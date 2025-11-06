# implementation of Pi-Flow for simple minded people like @cloneofsimo or like me.
# inspired and heavily based on: 
# Cloneofsimo minRF: https://github.com/cloneofsimo/minRF/tree/main
# PiFlow official: https://github.com/Lakonik/piFlow/tree/main

import argparse
import torch
import torch.nn.functional as F

class PiFlow:
    def __init__(self, student_model, teacher_model, NFE=4, DDIM_NFE=128, iter=2):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.iter = iter
        self.NFE = NFE
        self.DDIM_NFE = DDIM_NFE

        self.in_channels = student_model.in_channels
        self.input_size = student_model.input_size

    def pi(self, x_t, t, *args, A_s=None, mu_s=None, sigma_s=None, x_s=None, s=None, **kwargs):
        """
        Based on Gaussian-mixture model parameters from x_s, return v_t
        Please refer to Appendix F for more details. (notation: s <- t_src, t <- t_dst)

        GMM parameter tensor shapes (N, K, C, H, W):
        - A_s: (N, K, 1, H, W)
        - mu_s: (N, K, C, H, W)
        - sigma_s: (N, K, 1, 1, 1)
        - x_s: (N, K, C, H, W)
        - s: (N, 1, 1, 1, 1)

        Diffusion tensor shapes:
        - x_t: (N, C, H, W)
        - t: (N)
        """
        assert len(A_s.shape) == len(mu_s.shape) == len(sigma_s.shape) == len(x_s.shape) == len(s.shape) == 5, f"A_s: {A_s.shape}, mu_s: {mu_s.shape}, sigma_s: {sigma_s.shape}, x_s: {x_s.shape}, s: {s.shape}"
        assert len(x_t.shape) == 4, f"x_t: {x_t.shape}"
        assert len(t.shape) == 1, f"t: {t.shape}"

        if (t.squeeze() == s.squeeze()).all():
            return (A_s * mu_s).sum(dim=1, keepdim=True).squeeze(1)
        
        else:
            assert (s.squeeze() - t.squeeze() > 0).all(), f"s - t: {s.squeeze() - t.squeeze()}"

            # convert into q(x_0 | x_s)
            x_t = x_t[:, None, :, :, :]
            t = t[:, None, None, None, None]
            mu_x = x_s - s * mu_s
            sigma_x = s * sigma_s

            # convert into q(x_0 | x_t), Appendix F
            nu_x = s**2 * (1-t) * x_t - t**2 * (1-s) * x_s
            xi_x = s**2 * (1-t)**2 - t**2 * (1-s)**2
            denomi = xi_x * s**2 * t**2 + xi_x**2 * sigma_x**2

            # if t ~ s, assume v(x_t, t) ~ v(x_s, s) for numerical stability
            if (denomi < 1e-6).any():
                return (A_s * mu_s).sum(dim=1, keepdim=True).squeeze(1)

            a_t = A_s.log() - 1/2 * F.mse_loss(nu_x, xi_x*mu_x, reduction='none').sum(dim=2, keepdim=True) / denomi
            A_t = a_t.softmax(dim=1)

            mu_t = (sigma_x**2 * nu_x + s**2 * t**2 * mu_x) / (sigma_x**2 * xi_x + s**2 * t**2)
            x_0 = (A_t * mu_t).sum(dim=1, keepdim=True)
            v_t = (x_t.squeeze(1) - x_0.squeeze(1)) / t.squeeze(1)
            return v_t

    def forward_pi(self, z0=None, cond=None, iter=2, eps=1e-4):
        """
        Algorithm 2
        """
        b = z0.size(0)
        s = torch.randint(1, self.NFE + 1, (b,)).to(z0.device) / self.NFE
        sexp = s.view([b, *([1] * len(z0.shape[1:]))])

        z1 = torch.randn_like(z0)
        zs = (1 - sexp) * z0 + sexp * z1
        
        params = self.student_model(zs, s, cond)
        pi = lambda x_t, t, cond: self.pi(x_t, t, cond, **params)
        params_D = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in params.items()}
        pi_D = lambda x_t, t, cond: self.pi(x_t, t, cond, **params_D)

        loss = 0
        for iter_i in range(1, iter+1):
            t = s - 1 / self.NFE * (iter_i / iter)
            t = t.clamp(eps, 1).to(z0.device)
            with torch.no_grad():
                zt = self.from_s_to_t(zs, s, t, cond, pi_D)
                vt = self.teacher_model(zt, t, cond)
            vtheta = pi(zt, t, cond) 
            loss = loss + F.mse_loss(vt, vtheta)
        loss = loss / iter
        return loss

    def forward_pi_data_free(self, z0=None, cond=None, iter=2, eps=1e-4):
        """
        Algorithm 3
        """
        b = z0.size(0)
        s = torch.randint(1, self.NFE + 1, (b,)).to(z0.device) / self.NFE

        z1 = torch.randn_like(z0)
        zs = z1

        loss = 0
        for _ in range(self.NFE):
            # compute loss at time s
            params = self.student_model(zs, s, cond)
            pi = lambda x_t, t, cond: self.pi(x_t, t, cond, **params)
            params_D = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in params.items()}
            pi_D = lambda x_t, t, cond: self.pi(x_t, t, cond, **params_D)

            for iter_i in range(1, iter+1):
                t = s - 1 / self.NFE * (iter_i / iter)
                t = t.clamp(eps, 1).to(z0.device)
                with torch.no_grad():
                    zt = self.from_s_to_t(zs, s, t, cond, pi_D)
                    vt = self.teacher_model(zt, t, cond)
                vtheta = pi(zt, t, cond) 
                loss = loss + F.mse_loss(vt, vtheta)

            # update zs (s -> t)
            zs = self.from_s_to_t(zs, s, s-1/self.NFE, cond, pi_D).detach()
            s = s - 1 / self.NFE
        loss = loss / iter / self.NFE
        return loss

    @torch.no_grad()
    def sample_pi(self, x_s, cond):
        b = x_s.size(0)
        dt = 1 / self.NFE
        images = [x_s]
        
        s = torch.ones((b,)).to(x_s.device)
        for _ in range(self.NFE, 0, -1):
            t = s - dt
            params = self.student_model(x_s, s, cond)
            pi = lambda x_t, t, cond: self.pi(x_t, t, cond, **params)
            x_s = self.from_s_to_t(x_s, s, t, cond, pi)
            images.append(x_s)
            s = t
        return images

    @torch.no_grad()
    def from_s_to_t(self, x_s, s, t, cond, pi, NFE=64):
        """Analytical integration of ODE from s to t"""
        if (s == t).all():
            return x_s
        else:
            dt = (s - t) / NFE
            assert (dt > 0).all(), f"dt < 0"
            for _ in range(NFE):
                v_s = pi(x_s, s, cond)
                x_s = x_s - dt[:, None, None, None] * v_s
                s = s - dt
            return x_s

    # flow matching loss
    def forward_fm(self, z0, cond):
        b = z0.size(0)
        nt = torch.randn((b,)).to(z0.device)
        t = torch.sigmoid(nt)
        texp = t.view([b, *([1] * len(z0.shape[1:]))])
        z1 = torch.randn_like(z0)
        zt = (1 - texp) * z0 + texp * z1
        vtheta = self.teacher_model(zt, t, cond)
        loss = F.mse_loss(vtheta, z1 - z0)
        return loss.mean()


    # flow matching sampling
    @torch.no_grad()
    def sample_fm(self, x_t, cond, DDIM_NFE=128):
        b = x_t.size(0)
        dt = 1.0 / DDIM_NFE
        dt = torch.tensor([dt] * b).to(x_t.device).view([b, *([1] * len(x_t.shape[1:]))])
        images = [x_t]
        for i in range(DDIM_NFE, 0, -1):
            t = i / DDIM_NFE
            t = torch.tensor([t] * b).to(x_t.device)
            v_t = self.teacher_model(x_t, t, cond)
            x_t = x_t - dt * v_t
            images.append(x_t)
        return images

# helper function to convert list of tensors to gif
def tensors_to_gif(images):
    """Convert list of tensors to gif"""
    gif = []
    for image in images:
        image = image * 0.5 + 0.5
        image = image.clamp(0, 1)
        x_as_image = make_grid(image.float(), nrow=4)
        img = x_as_image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        gif.append(Image.fromarray(img))
    return gif

if __name__ == "__main__":
    # train class conditional RF on mnist.
    import os
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    try:
        import wandb
        is_wandb_available = True
    except ImportError:
        is_wandb_available = False

    from dit import DiT_Llama

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--NFE", type=int, default=4, help="number of NFE for student model")
    parser.add_argument("--K", type=int, default=8, help="number of Gaussian mixture components")
    parser.add_argument("--iter", type=int, default=2, help="number of analytical integration per training step")
    parser.add_argument("--data_free", action="store_true", help="use data-free training")
    args = parser.parse_args()

    result_dir = f"contents/{args.dataset}/NFE_{args.NFE}-K_{args.K}-iter_{args.iter}-data_free_{args.data_free}"
    weight_dir = f"weights/{args.dataset}/NFE_{args.NFE}-K_{args.K}-iter_{args.iter}-data_free_{args.data_free}"
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # for reproducibility
    import sys
    with open(sys.argv[0]) as f:
        code = f.read()
    with open(f"{result_dir}/piflow.txt", "w") as f:
        f.write(code)

    if args.dataset == "cifar":
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 3
        patch_size = 2
        teacher_model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10, patch_size=patch_size
        ).cuda()
        student_model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10, K=args.K, patch_size=patch_size
        ).cuda()

    elif args.dataset == "mnist":
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 1
        patch_size = 2
        teacher_model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10, patch_size=patch_size
        ).cuda()
        student_model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10, K=args.K, patch_size=patch_size
        ).cuda()

    rf = PiFlow(student_model, teacher_model, NFE=args.NFE, iter=args.iter)
    train_ds = fdatasets(root="./data", train=True, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)

    if is_wandb_available:
        wandb.init(project=f"piflow", name=f"{args.dataset}-NFE_{args.NFE}-K_{args.K}-iter_{args.iter}")

    # train teacher model (if there is no teacher model checkpoint)
    if os.path.exists(f"{weight_dir}/teacher.pth"):
        rf.teacher_model.load_state_dict(torch.load(f"{weight_dir}/teacher.pth"))
    else:
        epochs = 25
        optimizer = optim.Adam(rf.teacher_model.parameters(), lr=5e-4)
        for epoch in tqdm(range(epochs), desc="Training teacher model"):
            for i, (x, c) in enumerate(train_dl):
                x, c = x.cuda(), c.cuda()
                optimizer.zero_grad()
                loss = rf.forward_fm(x, c)
                loss.backward()
                optimizer.step()
                
                if is_wandb_available:
                    wandb.log({"teacher_loss": loss.item()})
        torch.save(rf.teacher_model.state_dict(), f"{weight_dir}/teacher.pth")

    # Generate samples from teacher model with NFE = 50
    x_T = torch.randn(16, channels, 32, 32, generator=torch.Generator().manual_seed(42)).cuda()
    cond = torch.arange(0, 16).cuda() % 10
    images = rf.sample_fm(x_T, cond, DDIM_NFE=50)
    gif = tensors_to_gif(images)
    gif[0].save(f"{result_dir}/sample_teacher_fm.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)
    gif[-1].save(f"{result_dir}/sample_teacher_fm_last.png")

    # initialize student model with teacher model weights
    with torch.no_grad():
        for name, param in rf.teacher_model.named_parameters():
            param.requires_grad = False
            if 'final_layer' not in name:
                rf.student_model.state_dict()[name].data.copy_(param.data)

    # train student model
    epochs = 100
    optimizer = optim.Adam(rf.student_model.parameters(), lr=5e-4)
    for epoch in tqdm(range(epochs), desc="Training student model"):
        for i, (x, c) in enumerate(train_dl):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            if args.data_free:
                loss = rf.forward_pi_data_free(x, c)
            else:
                loss = rf.forward_pi(x, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rf.student_model.parameters(), 1e2)
            optimizer.step()
            
            if is_wandb_available:
                wandb.log({"student_loss": loss.item()})
        
        # Generate samples from student model with NFE = 4
        rf.student_model.eval()
        with torch.no_grad():
            images = rf.sample_pi(x_T, cond)
            gif = tensors_to_gif(images)
            gif[0].save(f"{result_dir}/sample_{epoch}_pi.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)
            gif[-1].save(f"{result_dir}/sample_{epoch}_pi_last.png")
        rf.student_model.train()
