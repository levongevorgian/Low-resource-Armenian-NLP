# Google Cloud A100 GPU Setup Guide
## For Armenian Tokenizer Surgery Project — Goals 3 & 4

**Budget:** $300 free credits (new GCP account)
**Target:** 1x NVIDIA A100 40GB GPU, 85GB RAM, 200GB SSD
**Estimated cost:** ~$3.67/hour (on-demand) or ~$1.10/hour (spot)
**Usable hours at $300:** ~80 hours on-demand, ~270 hours spot

---

## Step 1: Create Google Cloud Account with $300 Free Credits

1. Go to https://cloud.google.com/free
2. Click **"Get started for free"**
3. Sign in with your Google account
4. Enter your country (Armenia) and agree to terms
5. **Billing step:** Enter a credit card — you will NOT be charged. Google requires it for identity verification only. The $300 credit is applied automatically.
6. You'll see a banner: "You have $300 in free trial credits"

**Important:** Free trial lasts 90 days. You won't be auto-charged after credits expire — Google requires explicit upgrade to a paid account.

---

## Step 2: Request GPU Quota (DO THIS FIRST — takes 5-30 minutes to approve)

New accounts have 0 GPU quota by default. You must request it.

1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. In the **Filter** bar, type: `GPUs (all regions)` 
3. Find the row **"GPUs (all regions)"** with limit **0**
4. Check the checkbox next to it
5. Click **"EDIT QUOTAS"** at the top
6. Set the new limit to **1**
7. Fill in your name, phone, and a brief justification: *"Academic research project — training an NLP model for Armenian language processing. Need 1 A100 GPU for approximately 10-20 hours."*
8. Submit the request

**Also request A100-specific quota:**
1. In the filter bar, type: `NVIDIA A100`
2. Find **"NVIDIA A100 GPUs"** for your target region (us-central1)
3. Request limit of **1** with the same justification

**Wait for approval email** (usually 5-30 minutes, sometimes up to 24 hours). You cannot create GPU VMs until this is approved.

---

## Step 3: Create the VM Instance

Once quota is approved:

### 3a. Navigate to VM creation
1. Go to: https://console.cloud.google.com/compute/instancesAdd
2. Or: Hamburger menu → Compute Engine → VM Instances → CREATE INSTANCE

### 3b. Configure the VM

**Name:** `armenian-tokenizer-surgery`

**Region & Zone:**
- Region: **us-central1** (cheapest for A100)
- Zone: **us-central1-a** or **us-central1-c** (whichever has A100 availability)
- If unavailable, try: us-east1-b, us-west1-b, europe-west4-a

**Machine configuration:**
- Machine family: **GPU**
- GPU type: **NVIDIA A100 40GB**
- Number of GPUs: **1**
- Machine type: **a2-highgpu-1g** (auto-selected: 12 vCPUs, 85 GB RAM)

**That 85GB RAM is optimal** — enough for the 0.5B model + tokenizer + training data with plenty of headroom. No need to customize.

### 3c. Boot disk
1. Click **"CHANGE"** under Boot disk
2. **Operating system:** Deep Learning on Linux
3. **Version:** Deep Learning VM with CUDA 12.4, M131 (or latest)
    - This comes with PyTorch 2.4+, CUDA, cuDNN pre-installed
4. **Boot disk type:** SSD persistent disk (Balanced persistent disk is also fine)
5. **Size:** **200 GB** (enough for corpus + model checkpoints + packages)
6. Click **SELECT**

### 3d. Firewall & Access
1. Check **"Allow HTTP traffic"** and **"Allow HTTPS traffic"** (optional, for Jupyter)
2. Under **Advanced options → Management → Availability policies:**
   - For cheaper runtime: Set **VM provisioning model** to **Spot** ($1.10/hr vs $3.67/hr)
   - Warning: Spot VMs can be preempted (shut down) with 30s notice. Save checkpoints frequently!
   - For critical training runs: Keep **Standard** (on-demand)

### 3e. Create
1. Click **"CREATE"**
2. Wait 1-2 minutes for the VM to boot
3. You'll see it listed with a green checkmark and an External IP

---

## Step 4: Connect to the VM

### Option A: Browser SSH (easiest)
1. In the VM instances list, click **"SSH"** button next to your VM
2. A browser terminal opens directly

### Option B: gcloud CLI (better for file transfers)
```bash
# Install gcloud CLI if needed: https://cloud.google.com/sdk/docs/install
gcloud compute ssh armenian-tokenizer-surgery --zone us-central1-a
```

### First login prompt
The Deep Learning VM will ask:
```
Would you like to install NVIDIA driver? [y/n]
```
Type **y** and wait ~2 minutes for driver installation.

Verify GPU is working:
```bash
nvidia-smi
```
You should see the A100 40GB listed.

---

## Step 5: Set Up the Environment

```bash
# The Deep Learning VM comes with conda and PyTorch pre-installed
# Verify PyTorch + CUDA
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Install additional packages
pip install transformers datasets sentencepiece protobuf accelerate peft bitsandbytes tqdm tabulate

# Create project directory
mkdir -p ~/project
cd ~/project
```

---

## Step 6: Upload Your Files to the VM

### From your MacBook terminal (not the VM):

```bash
# Upload the trained tokenizer models from this repository
gcloud compute scp models/tokenizers/hy_bpe_32k.model armenian-tokenizer-surgery:~/project/ --zone us-central1-a
gcloud compute scp models/tokenizers/hy_bpe_32k.vocab armenian-tokenizer-surgery:~/project/ --zone us-central1-a

# Upload the evaluation sample
gcloud compute scp data/sample/hy_sample_raw.txt armenian-tokenizer-surgery:~/project/ --zone us-central1-a

# Upload the Goal 2 results
gcloud compute scp results/goal2/goal2_eval_results.json armenian-tokenizer-surgery:~/project/ --zone us-central1-a

# Upload the Goal 3 notebook
gcloud compute scp goal_3_lightweight_adaptation/goal3_grafting.ipynb armenian-tokenizer-surgery:~/project/ --zone us-central1-a

# Upload the cleaned corpus from your local data archive (not committed here)
gcloud compute scp hy_clean.txt armenian-tokenizer-surgery:~/project/ --zone us-central1-a
```

**Alternative: Google Cloud Storage (faster for large files)**
```bash
# Create a bucket
gsutil mb gs://armenian-tokenizer-project

# Upload from MacBook
gsutil cp hy_clean.txt gs://armenian-tokenizer-project/

# Download on VM
gsutil cp gs://armenian-tokenizer-project/hy_clean.txt ~/project/
```

---

## Step 7: Run Jupyter on the VM (optional)

If you prefer Jupyter over terminal:

```bash
# On the VM
cd ~/project
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

Then on your MacBook, set up SSH tunnel:
```bash
gcloud compute ssh armenian-tokenizer-surgery --zone us-central1-a -- -L 8888:localhost:8888
```

Open http://localhost:8888 in your browser. Paste the token from the VM terminal.

---

## Step 8: Run Goal 3 & 4

```bash
cd ~/project
# Option 1: Run as notebook
jupyter notebook goal3_grafting.ipynb

# Option 2: Run as script (recommended for long training)
# Convert notebook to script first, or use the .py version
# Use tmux/screen so training continues if SSH disconnects:
tmux new -s training
python3 goal3_grafting.py
# Detach: Ctrl+B then D
# Reattach: tmux attach -t training
```

**CRITICAL: Use tmux or screen** — if your SSH connection drops during training, the process dies without it.

---

## Step 9: Download Results

After training completes:
```bash
# From your MacBook
gcloud compute scp armenian-tokenizer-surgery:~/project/results/* ./results/ --zone us-central1-a --recurse
```

---

## Step 10: STOP THE VM WHEN DONE

**This is the most important step.** A running A100 VM costs $3.67/hour even when idle.

```bash
# From your MacBook
gcloud compute instances stop armenian-tokenizer-surgery --zone us-central1-a

# Or in the Cloud Console: VM Instances → click the three dots → Stop
```

To restart later:
```bash
gcloud compute instances start armenian-tokenizer-surgery --zone us-central1-a
```

To delete permanently (after downloading all results):
```bash
gcloud compute instances delete armenian-tokenizer-surgery --zone us-central1-a
```

---

## Cost Summary

| Activity | Duration | Cost (on-demand) | Cost (spot) |
|----------|----------|-------------------|-------------|
| Goal 3: Grafting experiments | ~1-2 hours | $4-7 | $1-2 |
| Goal 4: LoRA fine-tuning | ~3-6 hours | $11-22 | $3-7 |
| Goal 5: Evaluation | ~1 hour | $4 | $1 |
| File transfers, setup | ~1 hour | $4 | $1 |
| **Total estimated** | **~6-10 hours** | **$22-37** | **$6-11** |

Well within the $300 free credit budget. You could run this project 10x over.

---

## Troubleshooting

**"Quota GPUS_ALL_REGIONS exceeded"**
→ Your GPU quota request hasn't been approved yet. Check email or re-submit.

**"The zone does not have enough resources available"**
→ Try a different zone in the same region, or try spot VMs which have different availability.

**"NVIDIA driver not found"**
→ Run `sudo /opt/deeplearning/install-driver.sh` on the VM.

**SSH connection drops during training**
→ Always use `tmux` or `screen`. See Step 8.

**Running out of disk space**
→ Check with `df -h`. If needed, resize the disk in Cloud Console (VM → Edit → Boot disk → increase size), then on the VM: `sudo resize2fs /dev/sda1`
