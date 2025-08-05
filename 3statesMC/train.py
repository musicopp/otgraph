import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import *
from nn import *
from ot import *
torch.manual_seed(2010)


if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.init()
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
T = 1
# ================= Hyperparameters ====================
default_lr = 5e-6  # learning rate for net
default_iterations = 2000000 + 1
start_iteration = 0
activation_f = "tanh"  # ReLU tanh SiLU
# =====================================================


# ---------------------------
# Main training script with dynamic coefficient update
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geodesic path solver on a manifold")
    parser.add_argument(
        "--lr", type=float, default=default_lr, help="Learning rate for net_x"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=default_iterations,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--load_model_rho",
        type=str,
        default=None,
        help="Path to load pretrained net_x model",
    )
    parser.add_argument(
        "--load_model_psi",
        type=str,
        default=None,
        help="Path to load pretrained net_p model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to load pretrained net_p model",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="12",
        help="Name of the endpoints",
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        default="1",
        help="Name of the folder to save the model",
    )
    args = parser.parse_args()

    lr, iterations = args.lr, args.iterations

    # Initialize boundary conditions
    # Boundary conditions
    x_1 = torch.tensor([[0.45*4.5, 0.45*3.,0.1*9./4.]]).to(device)
    x_2 = torch.tensor([[0.1*4.5, 0.45*3.,0.45*9./4.]]).to(device)
    x_3 = torch.tensor([[0.45*4.5, 0.1*3.,0.45*9./4.]]).to(device)

    if args.start == "12":
        x_start = x_1
        x_end = x_2
        start_invert_coeff = invert_coeff1
        end_invert_coeff = invert_coeff2
    if args.start == "23":
        x_start = x_2
        x_end = x_3
        start_invert_coeff = invert_coeff2
        end_invert_coeff = invert_coeff3
    if args.start == "31":
        x_start = x_3
        x_end = x_1
        start_invert_coeff = invert_coeff3
        end_invert_coeff = invert_coeff1
    pt_t_init = torch.tensor([[0.0]]).to(device)
    pt_t_end = torch.tensor([[1.0]]).to(device)
    # ReLU tanh SiLU
    if args.model == "sine":
         net_rho = barycentersine(1,3,activation_fn=activation_f,transform_fn=None,start_invert_coeff=start_invert_coeff,end_invert_coeff=end_invert_coeff).to(device)
    if args.model == "poly":
         net_rho = polyNN(1,3,activation_f,projection_p).to(device)
    if args.model == "basic":
         net_rho = basicNN(1,3,activation_f,projection_p).to(device)
    if args.model == "barycenter":
         net_rho = barycenterNN(1,3,activation_f).to(device)
    if args.model == "barycenteradd":
         net_rho = barycenteraddNN(1,3,activation_fn=activation_f,transform_fn=None,invert_coeff=start_invert_coeff).to(device)
    net_psi = basicNN(1,3,activation_f).to(device)
    if args.load_model_rho:
        net_rho.load_state_dict(torch.load(args.load_model_rho, map_location=device))
        print("Loaded net_x from", args.load_model_rho)
    if args.load_model_psi:
        net_psi.load_state_dict(torch.load(args.load_model_psi, map_location=device))
        print("Loaded net_p from", args.load_model_psi)

    
    loss_fn = nn.MSELoss()

    optimizer_rho = torch.optim.SGD(net_rho.parameters(), lr=lr)
    optimizer_psi = torch.optim.SGD(net_psi.parameters(), lr=lr)

    net_rho.train()
    net_psi.train()

    # Initialize coefficients for loss components (they must sum to 1)
    f1, f2 = 0.50, 0.50

    folder_name = args.folder_name
    os.makedirs(f"runs/ot/{folder_name}", exist_ok=True)
    os.makedirs(f"ot_model/{folder_name}", exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/ot/{folder_name}", comment="")

    for epoch in tqdm(range(start_iteration, iterations)):
        optimizer_rho.zero_grad()
        optimizer_psi.zero_grad()

        mse_init = loss_fn(net_rho(pt_t_init), x_start)
        mse_end = loss_fn(net_rho(pt_t_end), x_end)

        t_collocation = np.random.uniform(0.0, T, size=(5000, 1)).astype(np.float32)
        pt_t_collocation = torch.from_numpy(t_collocation).to(device)
        pt_t_collocation.requires_grad_()  # Ensure t requires grad
        pde1, pde2 =  pde(pt_t_collocation,net_rho,net_psi,loss_fn)

        loss = f1 * pde1 + f2 * pde2
        if args.model == "basic" or args.model == "barycenter":
            loss += mse_init + mse_end
        if args.model == 'barycenteradd':
            bc1 = 2.0
            loss += bc1*mse_end
        loss.backward()
        optimizer_rho.step()
        optimizer_psi.step()

        with torch.no_grad():
            if epoch % 1000 == 0 and epoch > 0:
                print(
                    f"Epoch:{epoch} mse_init:{mse_init.item():.6f} mse_end:{mse_end.item():.6f} "
                    f"pde1:{pde1.item():.6f} pde2:{pde2.item():.6f} Combined Loss:{loss.item():.6f}"
                )
        writer.add_scalar("Loss mse_init", mse_init, epoch)
        writer.add_scalar("Loss mse_end", mse_end, epoch)
        writer.add_scalar("Loss PDE 1", pde1, epoch)
        writer.add_scalar("Loss PDE 2", pde2, epoch)
        writer.add_scalar("Loss Combined", loss, epoch)
        #Dynamic coefficient update every 5000 epochs.
        if epoch % 1000 == 0 and epoch > 10000:
            loss_vals = [
                pde1.item(),
                pde2.item(),
            ]
            idx_max, idx_min = np.argmax(loss_vals), np.argmin(loss_vals)
            if loss_vals[idx_max] > 10 * loss_vals[idx_min]:
                coeffs = [f1, f2]
                transfer = 0.5 * coeffs[idx_min]
                coeffs[idx_min] *= 0.5
                coeffs[idx_max] += transfer
                f1, f2 = coeffs
                print(
                    f"Dynamic update at epoch {epoch}:  f1={f1:.4f}, f2={f2:.4f}"
                )
                writer.add_scalar("Coefficient f1", f1, epoch)
                writer.add_scalar("Coefficient f2", f2, epoch)
        # Save checkpoints every 2000 epochs.
        if epoch % 10000 == 0 and epoch > 0:
            torch.save(net_rho.state_dict(), f"ot_model/{folder_name}/rho_{epoch}.pth")
            torch.save(net_psi.state_dict(), f"ot_model/{folder_name}/psi_{epoch}.pth")

# example command
# $ python train.py --model sine --folder_name sine --start 12
# srun -c 40 --gres=gpu:1 --partition=batch python train.py --model sine --folder_name sine_23 --start 23