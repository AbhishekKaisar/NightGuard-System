# Contributing Guide — NightGuard System

## For Team Members

### Step 1: Clone the repo
```bash
git clone https://github.com/<username>/NightGuard-System.git
cd NightGuard-System
```

### Step 2: Create your branch
```bash
git checkout -b feature/<your-module-name>
```

Use one of these branch names:
- `feature/enhancement` — Low-Light Enhancement Lead
- `feature/face-detection` — Face Detection Lead
- `feature/human-detection` — Human Detection Lead
- `feature/vehicle-detection` — Vehicle Detection Lead

### Step 3: Add your work
Place your files in the correct folder:

| Module | Folder |
|--------|--------|
| Low-Light Enhancement | `modules/enhancement/` |
| Face Detection | `modules/face_detection/` |
| Human Detection | `modules/human_detection/` |
| Vehicle Detection | `modules/vehicle_detection/` |

Also copy your notebook (`.ipynb`) into the `notebooks/` folder.

### Step 4: Push and create a Pull Request
```bash
git add .
git commit -m "Add <module-name> module"
git push origin feature/<your-module-name>
```

Then go to GitHub and open a Pull Request to merge into `main`.

### Important Notes
- Do NOT push datasets or model weight files (`.pt`, `.h5`) — they are in `.gitignore`
- Keep your code inside your assigned module folder
- Add a brief README.md in your module folder describing your approach
