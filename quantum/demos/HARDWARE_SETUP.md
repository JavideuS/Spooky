# IBM Quantum Hardware Setup Guide

This guide walks you through setting up access to IBM Quantum hardware for running quantum pathfinding on real quantum computers.

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection
- Email address for IBM Quantum account

## 🚀 Step-by-Step Setup

### Step 1: Create IBM Quantum Account

1. **Visit IBM Quantum**: https://quantum.ibm.com/
2. **Sign up** for a free account (or log in if you have one)
3. **Verify your email** address

**Free tier includes**:
- Access to real quantum hardware
- Limited queue priority
- Sufficient for learning and small experiments

### Step 2: Get Your API Token

1. **Log in** to IBM Quantum Platform
2. **Click your profile icon** (top right)
3. **Navigate to**: Account Settings → API Token
4. **Copy your token** (looks like: `abc123def456...`)

**⚠️ Keep your token secret!** Don't commit it to git or share it publicly.

### Step 3: Configure Your Credentials

The recommended way is to save your credentials locally using Qiskit. This only needs to be done once.

#### Option A: Save Account (Recommended)

Run this Python snippet (replacing `your_token_here` with your actual token):

```bash
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum', token='your_token_here', overwrite=True)"
```

This saves your token to `~/.qiskit/qiskit-ibm.json`.

Remember to never commit your token to git or share it publicly.

#### Option B: Environment Variable

Alternatively, you can set an environment variable for temporary access:

**Linux/macOS**:
```bash
export IBM_QUANTUM_TOKEN='your_token_here'
```

**Windows (PowerShell)**:
```powershell
$env:IBM_QUANTUM_TOKEN='your_token_here'
```

To make it permanent:
```powershell
[System.Environment]::SetEnvironmentVariable('IBM_QUANTUM_TOKEN', 'your_token_here', 'User')
```

#### Option C: Configuration File

Create `~/.qiskit/qiskitrc`:
```ini
[ibm-quantum]
token = your_token_here
url = https://auth.quantum-computing.ibm.com/api
```


### Step 4: Install Required Packages

```bash
pip install qiskit qiskit-ibm-runtime pennylane pennylane-qiskit
```

**Verify installation**:
```bash
python -c "import qiskit; print(qiskit.__version__)"
```

### Step 5: Test Your Connection

Create a test script `test_ibm_connection.py`:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
import os

# Test connection
try:
    # This will check saved accounts OR environment variables
    service = QiskitRuntimeService()
    backends = service.backends()
    
    print(f"\n✓ Connected to IBM Quantum!")
    print(f"✓ Available backends: {len(backends)}")
    
    print("\nYour available quantum computers:")
    for backend in backends[:5]:  # Show first 5
        print(f"  - {backend.name}: {backend.num_qubits} qubits")
    
    print("\n✅ Setup complete! You're ready to run demo 05.")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Check your token is correct")
    print("  2. Verify internet connection")
    print("  3. Try regenerating your token on IBM Quantum")

```

**Run it**:
```bash
python test_ibm_connection.py
```

**Expected output**:
```
✓ Token found
✓ Connected to IBM Quantum!
✓ Available backends: 15

Your available quantum computers:
  - ibm_fez: 156 qubits
  - ibm_marrakesh: 156 qubits
  - ibm_torino: 133 qubits
  ...

✅ Setup complete! You're ready to run demo 05.
```

## 🎯 Running the Hardware Demo

Once setup is complete:

```bash
cd quantum
python demos/05_hardware_qiskit.py
```

**What to expect**:
1. Credential check (should pass ✓)
2. Problem creation (3x3 grid, very small)
3. Job submission to IBM Quantum
4. **Moderate wait** (queue time: few seconds to minutes)
5. Results from real quantum hardware!

**Check queue status**: https://quantum.ibm.com/services/resources

## 🐛 Troubleshooting

### "Too many qubits" error

## 📊 Monitoring Your Jobs

### Via Web Interface

1. Go to: https://quantum.ibm.com/jobs
2. View all your submitted jobs
3. Check status, queue position, and results

### Via Code

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
jobs = service.jobs(limit=5)

for job in jobs:
    print(f"Job {job.job_id()}: {job.status()}")
```

## ❓ FAQ

**Q: Do I need to pay for IBM Quantum?**  
A: No, the free tier is sufficient for demos and learning.

**Q: How long will my job take?**  
A: Anywhere from seconds to minutes, depending on queue.

**Q: Can I cancel a queued job?**  
A: Yes, via the web interface or API.

**Q: What if hardware is down?**  
A: Check the status page and try again later.

**Q: Can I use other quantum providers?**  
A: Yes! The system supports any Pennylane-compatible backend or D-Wave.

**Ready to run on real quantum hardware? 🚀**

Return to [demos README](README.md) and run `05_hardware_qiskit.py`!
