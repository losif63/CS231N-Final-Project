# CS231N Final Project

The goal of this project is to predict difficulty from a playthrough video of a Geometry Dash level.

## Data Collection

To run the level metadata collection script (`geometry_dash/collect.js`), first initialize an npm project and install the required dependency:

```bash
npm init -y
npm install gd.js
```

Then run the script:

```bash
node geometry_dash/collect.js
```