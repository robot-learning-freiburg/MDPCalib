# MDPCalib
[**arXiv**](https://arxiv.org) | [**Website**](http://calibration.cs.uni-freiburg.de/) | [**Video**](https://youtu.be/L1MwAenzd6g)

This repository is the official implementation of the paper:

> **Automatic Target-Less Camera-LiDAR Calibration from Motion and Deep Point Correspondences**
>
> [Kürsat Petek](http://www2.informatik.uni-freiburg.de/~petek/)&ast;, [Niclas Vödisch](https://vniclas.github.io/)&ast;, [Johannes Meyer](http://www2.informatik.uni-freiburg.de/~meyerjo/), [Daniele Cattaneo](https://rl.uni-freiburg.de/people/cattaneo), [Abhinav Valada](https://rl.uni-freiburg.de/people/valada), and [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard/). <br>
> &ast;Equal contribution. <br> 
> 
> *arXiv preprint*, 2024

<p align="center">
  <img src="./assets/mdpcalib_overview.png" alt="Overview of MDPCalib approach" width="800" />
</p>

If you find our work useful, please consider citing our paper:
```
@article{petek2024mdpcalib,
  title={Automatic Target-Less Camera-LiDAR Calibration from Motion and Deep Point Correspondences},
  author={Petek, Kürsat and Vödisch, Niclas and Meyer, Johannes and Cattaneo, Daniele and Valada, Abhinav and Burgard, Wolfram},
  journal={arXiv preprint},
  year={2024}
}
```


## 📔 Abstract

Sensor setups of robotic platforms commonly include both camera and LiDAR as they provide complementary information. However, fusing these two modalities typically requires a highly accurate calibration between them. In this paper, we propose MDPCalib which is a novel method for camera-LiDAR calibration that requires neither human supervision nor any specific target objects. Instead, we utilize sensor motion estimates from visual and LiDAR odometry as well as deep learning-based 2D-pixel-to-3D-point correspondences that are obtained without in-domain retraining. We represent the camera-LiDAR calibration as a graph optimization problem and minimize the costs induced by constraints from sensor motion and point correspondences. In extensive experiments, we demonstrate that our approach yields highly accurate extrinsic calibration parameters and is robust to random initialization. Additionally, our approach generalizes to a wide range of sensor setups, which we demonstrate by employing it on various robotic platforms including a self-driving perception car, a quadruped robot, and a UAV.


## 👩‍💻 Code

We will release the code upon the acceptance of our paper.


## 👩‍⚖️  License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
For any commercial purpose, please contact the authors.


## 🙏 Acknowledgment

This work was funded by the German Research Foundation (DFG) Emmy Noether Program grant No 468878300 and an academic grant from NVIDIA.
<br><br>
<p float="left">
  <a href="https://www.dfg.de/en/research_funding/programmes/individual/emmy_noether/index.html"><img src="./assets/dfg_logo.png" alt="DFG logo" height="100"/></a>
</p>
