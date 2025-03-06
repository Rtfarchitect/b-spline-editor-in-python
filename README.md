# Spline Editor

## Description

A Python-based interactive b-spline editor designed for manipulating surfaces. It utilizes advanced geometry processing libraries for intuitive editing and rendering of splines.

## Features

Left-click and drag: Rotate view

##

&#x20; Right-click and drag: Move control points

&#x20; Scroll: Zoom in/out

&#x20; Key 1: Toggle control grid visibility

&#x20; Key 2: Toggle surface visibility

&#x20; Key A: Add new B-spline patch (copy of current)

&#x20; Key N: Create new blank B-spline patch

&#x20; Key TAB: Switch to next patch

&#x20; Key P: Switch to previous patch

&#x20; Key ESC: Exit



### Python Libraries

- `polyscope`
- `glfw`
- `numpy`
- `scipy`
- `pygeodesic`
- `ipywidgets`
- `pyopengl`

### Conda-specific installations

Run the following commands to install additional dependencies:

```bash
conda install -c conda-forge igl
conda install -c conda-forge meshplot
```

## Installation

1. Clone the repository:

```bash
git clone <your_repository_url>
```

2. Create and activate the conda environment:

```bash
conda create -n spline python=3.9
conda activate spline
```

3. Install dependencies:

```bash
pip install -r requirements.txt
conda install -c conda-forge igl meshplot
```

## Usage

Run the spline editor with:

```bash
python main.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

