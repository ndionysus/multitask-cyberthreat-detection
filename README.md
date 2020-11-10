# Towards end-to-end Cyberthreat Detection from Twitter using Multi-Task Learning 

This repository holds the data, source code and resulting model weights for the paper "Towards end-to-end Cyberthreat Detection from Twitter using Multi-Task Learning" to be presented at IJCNN 2020.

This repository is still a work in progress.

TO DO: </br>
  [x] - Fix any errors with the initial commit </br>
  [x] - Save weights and conf </br>
  [ ] - Load weights and conf </br>
  [ ] - Complete README file. </br>
  &nbsp;&nbsp;&nbsp;  [x] - Add required packages and version numbers.</br>
  &nbsp;&nbsp;&nbsp;  [ ] - Add general usability, such as train, save and load models.</br>
  &nbsp;&nbsp;&nbsp; [ ] - Add examples to evaluate external data. </br>
  [ ] - General housekeeping to remove unused functions and add comments </br>

# Requierments
torch == 1.5.1 </br>
pandas == 1.0.5 </br>
pytorch-crf == 0.7.2 </br>

# Citation

```
@INPROCEEDINGS{9207159,
  author={N. {Dionísio} and F. {Alves} and P. M. {Ferreira} and A. {Bessani}},
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Towards end-to-end Cyberthreat Detection from Twitter using Multi-Task Learning}, 
  year={2020},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IJCNN48605.2020.9207159}}
```

# References

CRF module : https://github.com/kmkurn/pytorch-crf

@inproceedings{paszke2017automatic,
 title={Automatic Differentiation in {PyTorch}},
 author={Paszke, Adam and Gross, Sam and Chintala, Soumith and Chanan, Gregory and Yang, Edward and DeVito, Zachary and Lin, Zeming and Desmaison, Alban and Antiga, Luca and Lerer, Adam},
 booktitle={NIPS Autodiff Workshop},
 year={2017}
}
