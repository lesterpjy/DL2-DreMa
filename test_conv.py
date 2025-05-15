# test_conv.py
import torch

torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.deterministic = False

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    try:
        # Input that causes the error: (8, 3, 256, 256)
        # Conv1 in CLIP: 3 input channels, 32 output channels, kernel 3, stride 2, padding 1
        inp = torch.randn(8, 3, 256, 256, device=device, dtype=torch.float32)
        conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False).to(device)
        
        print("Running conv...")
        out = conv(inp)
        print(f"Output shape: {out.shape}") # Expected: [8, 32, 128, 128]
        print("Conv2D successful!")

    except RuntimeError as e:
        print(f"ERROR during Conv2D: {e}")
        if "no valid convolution algorithms available in CuDNN" in str(e):
            print("REPRODUCED THE CuDNN ERROR in minimal script!")
        else:
            print("Got a different RuntimeError in minimal script.")
    except Exception as e:
        print(f"An unexpected error occurred in minimal script: {e}")
else:
    print("CUDA not available, skipping Conv2D test.")
