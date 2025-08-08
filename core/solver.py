import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
from collections import OrderedDict
import jittor as jt
import jittor.nn as nn

jt.flags.use_cuda = 1

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics

def adv_loss(logits, target) -> jt.Var:
    targets = jt.full_like(logits, target)
    loss = nn.binary_cross_entropy_with_logits(logits, targets)
    return loss

def r1_reg(d_out, x_in):
    batch_size = x_in.shape[0]
    x_in.start_grad()
    grad = jt.grad(d_out.sum(), x_in, retain_graph=True)[0]
    grad2 = grad.pow(2)
    reg = 0.5 * grad2.reshape(batch_size, -1).sum(1).mean(0)
    return reg


def moving_average(model, model_test, decay=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = utils.lerp(param.data, param_test.data, decay)


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None, itr=None, LogD=None):
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    if z_trg is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg

    return loss, Munch(real=loss_real, fake=loss_fake, reg=loss_reg)


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None, itr=None, LogG=None):
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs

    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = jt.mean(jt.abs(s_pred - s_trg))

    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)

    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)

    loss_ds = jt.mean(jt.abs(x_fake - x_fake2))

    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)

    loss_cycle = jt.mean(jt.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cycle
    return loss, Munch(adv=loss_adv, sty=loss_sty, ds=loss_ds, cycle=loss_cycle)


class Solver(jt.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.nets, self.nets_ema = build_model(args)

        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = jt.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, 'nets_{:06d}.pth'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, 'nets_ema_{:06d}.pth'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, 'optims_{:06d}.pth'), **self.optims)]

        else:
            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, 'nets_ema_{:06d}.pth'), data_parallel=True, **self.nets_ema)]

        for name, network in self.named_children():
            if 'ema' not in name:
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            inputs = next(fetcher)
            x_real = inputs.x_src
            y_org = inputs.y_src
            x_ref = inputs.x_ref
            x_ref2 = inputs.x_ref2
            y_trg = inputs.y_ref
            z_trg = inputs.z_trg
            z_trg2 = inputs.z_trg2
            masks = None

            d_loss, d_losses_latent = compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks, itr=i)
            optims['discriminator'].step(d_loss, retain_graph=True)

            d_loss, d_losses_ref = compute_d_loss(nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks, itr=i)
            optims['discriminator'].step(d_loss, retain_graph=True)

            g_loss, g_losses_latent = compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=(z_trg, z_trg2), masks=masks, itr=i)
            optims['generator'].step(g_loss, retain_graph=True)
            optims['mapping_network'].step(g_loss, retain_graph=True)
            optims['style_encoder'].step(g_loss, retain_graph=True)

            g_loss, g_losses_ref = compute_g_loss(nets, args, x_real, y_org, y_trg, x_refs=(x_ref, x_ref2), masks=masks, itr=i)
            optims['generator'].step(g_loss, retain_graph=True)

            moving_average(nets.generator, nets_ema.generator, decay=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, decay=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, decay=0.999)

            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)


            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i + 1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i + 1)

            # save model checkpoints
            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)

            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, step=i+1, mode='latent')
                calculate_metrics(nets_ema, args, step=i+1, mode='reference')

    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema

        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        # Fetch source and reference images
        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print(f'Working on {fname}...')
        utils.translate_using_reference(
            nets_ema, args, src.x, ref.x, ref.y, fname)

    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema

        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=args.resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=args.resume_iter, mode='reference')
