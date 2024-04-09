# ACID DDSP
TBD

<hr>
<h2>Instructions for Reproducibility</h2>

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
