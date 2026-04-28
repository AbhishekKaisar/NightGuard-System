# Contributing Guide — NightGuard System

## Team Members & Assignments

| Name | ID | Branch Name | Module Folder |
|------|----|-------------|---------------|
| Anindya Saha Ani | 2221105042 | `feature/enhancement` | `modules/enhancement/` |
| Midhat Bin Shazzad | 2222560642 | `feature/face-detection` | `modules/face_detection/` |
| Abhishek Kaisar Abhoy | 2221140042 | `feature/human-detection` | `modules/human_detection/` |
| Maisha Tabassum | 2222728042 | `feature/vehicle-detection` | `modules/vehicle_detection/` |

---

## Step-by-Step Guide

### Step 1: Clone the repo
```bash
git clone https://github.com/AbhishekKaisar/NightGuard-System.git
cd NightGuard-System
```

### Step 2: Create your branch
Use the branch name from the table above.
```bash
git checkout -b feature/<your-module-name>
```

**Example (for Anindya):**
```bash
git checkout -b feature/enhancement
```

### Step 3: Add your work
Put your files in the correct folder (check the table above).

**Also** copy your Jupyter notebook (`.ipynb`) into the `notebooks/` folder.

### Step 4: Push and create a Pull Request
```bash
git add .
git commit -m "Add <module-name> module"
git push origin feature/<your-module-name>
```

Then go to GitHub → your branch → click **"Compare & pull request"** → submit.

---

## Important Rules
- Do NOT push datasets or model weight files (`.pt`, `.h5`) — they are ignored by `.gitignore`
- Only add files inside your assigned module folder
- Each module folder has a README — update it with your actual approach and results
