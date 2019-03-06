# Single-View-Place-Recognition

Single-View Place Recognition under Seasonal Changes [[arXiv]](https://arxiv.org/pdf/1808.06516.pdf)

Authors: Daniel Olid, [Jose M. Facil](http://webdiis.unizar.es/~jmfacil) and [Javier Civera](http://webdiis.unizar.es/~jcivera)

Project Website: [[project web]](http://webdiis.unizar.es/~jmfacil/pr-nordland/)

Partitioned Nordland Dataset: [[download]](http://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset)

This work was continued with a Multi-View version of our method, please find our latest work at [Condition-Invariant Multi-View Place Recognition
](http://webdiis.unizar.es/~jmfacil/cimvpr/)

## How to use:
We recommend the use of a virtual enviroment for the use of this project. ([pew](https://github.com/berdario/pew))
```bash
$ pew new venvname -p python3 # replace venvname with your prefered name (it also works with python 2.7)
```
### 1. Install Caffe
Install [Caffe](https://github.com/BVLC/caffe) following the instructions.

Please notice if you are using a virtual enviroment to install the python requirements inside the virtual enviroment:
```bash
$ pew in venvname
(venvname)$ pip install $req # req variable contains python module to be installed
```

And add Caffe python module to the PATH in the virtual enviroment:
```bash
(venvname)$ pew add caffe/python # replace the path with the path to your caffe repo
```

### 2. Install Remaining Dependences
Install remaining dependences:
```bash
(venvname)$ for req in $(cat requirements.txt); do pip install $req; done
```

### 3. Run Test
To run our demo please run:
```bash
(venvname)$ python test_norland.py --help
```
Note: This version runs with the downsampled version of Partitioned Nordland.
## Contact
You can find my contact information in my [Personal Website](http://webdiis.unizar.es/~jmfacil/)
## License and Citation
This software is under GNU General Public License Version 3 (GPLv3), please see [GNU License](http://www.gnu.org/licenses/gpl.html)

For commercial purposes, please contact the authors.

Please cite our paper if it helps your research:

  ```bibtex
  @article{olid2018single,
    title={Single-View Place Recognition under Seasonal Changes},
    author={Olid, Daniel and F{\'a}cil, Jos{\'e} M and Civera, Javier},
    journal={arXiv preprint arXiv:1808.06516},
    year={2018}
  }
  ```
## Disclaimer

This site and the code provided here are under active development. Even though we try to only release working high quality code, this version might still contain some issues. Please use it with caution.
