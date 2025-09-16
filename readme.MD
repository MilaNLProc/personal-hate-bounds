# Personalization up to a Point

This repo contains the code for the _Personalization up to a point_ paper.

# File catalog

## Modules

Directory: `modules/`

Modules provide the basic functions for data and model manipulation, with no entrypoint.

## Notebooks

Directory: `notebooks/`

Notebooks provide entrypoints for all the phases of the project (data exploration, dataset building, model training, evaluation). The three main ones are:
- `build_stratified_dataset.ipynb`: builds the stratified train-test split for the DTC (also called "Kumar") dataset. The resulting splits are then used for model training.
- `majority_vote_model.ipynb`: trains the majority vote (`Maj`) model.
- `sepheads_model_training.ipynb`: trains the personalized SepHeads (`Sep`) model.

## Scripts

Directory: `scripts/`

Scripts for model training, equivalent to the corresponding notebooks but in a form runnable through e.g. on a cluster via a job scheduler.

# License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
