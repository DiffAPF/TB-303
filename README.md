<div align="center">
<h1>Differentiable All-pole Filters for Time-varying Audio Systems</h1>

<p>
    <a href="https://yoyololicon.github.io/" target=”_blank”>Chin-Yun Yu</a>,
    <a href="https://christhetr.ee/" target=”_blank”>Christopher Mitcheltree</a>,
    <a href="https://www.linkedin.com/in/alistair-carson-a6178919a/" target=”_blank”>Alistair Carson</a>,
    <a href="https://www.acoustics.ed.ac.uk/group-members/dr-stefan-bilbao/" target=”_blank”>Stefan Bilbao</a>,
    <a href="https://www.eecs.qmul.ac.uk/~josh/" target=”_blank”>Joshua D. Reiss</a>, and
    <a href="https://www.eecs.qmul.ac.uk/~gyorgyf/about.html" target=”_blank”>György Fazekas</a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2404.07970-b31b1b.svg)](https://arxiv.org/abs/2404.07970)
[![Listening Samples](https://img.shields.io/badge/%F0%9F%94%8A%F0%9F%8E%B6-Listening_Samples-blue)](https://diffapf.github.io/web/)
[![Plugins](https://img.shields.io/badge/neutone-Plugins-blue)](https://diffapf.github.io/web/index.html#plugins)
[![License](https://img.shields.io/badge/License-MPL%202.0-orange)](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)

<h2>Time-varying Subtractive Synthesizer (<em>Roland TB-303 Bass Line</em>) Experiments</h2>
</div>

<h3>Instructions for Reproducibility</h3>

<ol>
    <li>Clone this repository and open its directory.</li>
    <li>Initialize and update the submodules (<code>git submodule update --init --recursive</code>).</li>
    <li>
    Install the requirements using <br><code>conda env create --file=conda_env_cpu.yml</code> or <br>
    <code>conda env create --file=conda_env.yml</code><br> for GPU acceleration.<br>
    <code>requirements_pipchill.txt</code> and <code>requirements_all.txt</code> are also provided as references, but are not needed when using the <code>conda_env.yml</code> files.
    </li>
    <li>The source code can be explored in the <code>acid_ddsp/</code> directory.</li>
    <li>All models from the paper can be found in the <code>models/</code> directory.</li>
    <li>All eval results from the paper can be found in the <code>eval/</code> directory.</li>
    <li>All <a href="https://neutone.ai" target=”_blank”>Neutone</a> files for running the models and the acid synth implementations as a VST in a DAW can be found in the <code>neutone/</code> directory.</li>
    <li>Create an out directory (<code>mkdir out</code>).</li>
    <li>
    All models can be evaluated by modifying and running <code>scripts/test.py</code>.<br>
    Make sure your <code>PYTHONPATH</code> has been set correctly by running a command like<br>
    <code>export PYTHONPATH=$PYTHONPATH:BASE_DIR/acid_ddsp/</code>,<br>
    <code>export PYTHONPATH=$PYTHONPATH:BASE_DIR/torchlpc/</code>, and<br>
    <code>export PYTHONPATH=$PYTHONPATH:BASE_DIR/fadtk/</code>.
    </li>
    <li>
    CPU benchmark values can be obtained by running <code>scripts/benchmark.py</code>.<br>
    These will vary depending on your computer.
    </li>
    <li>
    (Optional) All models can be trained by modifying <code>configs/abstract_303/train.yml</code> and running <code>scripts/train.py</code>.<br>
    Before training, <code>scripts/preprocess_data.py</code> should be run to create the dataset. 
    </li>
    <li>
    (Optional) Custom <a href="https://neutone.ai" target=”_blank”>Neutone</a> models can be exported by modifying and running <code>scripts/export_neutone_models.py</code> or <code>scripts/export_neutone_synth.py</code>.
    </li>
    <li>
    The source code is currently not documented, but don't hesitate to open an issue if you have any questions or comments.
    </li>
</ol>

## Citation

```bibtex
@inproceedings{ycy2024diffapf,
  title={Differentiable All-pole Filters for Time-varying Audio Systems},
  author={Chin-Yun Yu and Christopher Mitcheltree and Alistair Carson and Stefan Bilbao and Joshua D. Reiss and György Fazekas},
  booktitle={International Conference on Digital Audio Effects (DAFx)},
  year={2024}
}
```
