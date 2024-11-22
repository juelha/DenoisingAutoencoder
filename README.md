# DenoisingAutoencoder
- Demo of a Denoising Autoencoder (DAE) learning vector fields for one-dimensional manifolds.

- Programmed for the seminar 'Concepts in Deep Learning' at the University of Osnabrück (2024/2025) 

- By the group DataSippers

### Manifolds
In this demo, we will see how a denoising autoencoder will learn the vector field for three examples of one-dimensional manifolds: a line, a circle and a spiral.

*But a circle is not one but two-dimensional!* you may correctly say. That is where the concept of **manifolds** comes in. In mathematics, a manifold is a topological space that locally resembles
euclidean space. For example, we all know the earth is a sphere (I hope) but if we stand in an open field, it looks like we are standing on a plane, a two-dimensional space. 

For our circle let's imagine we are a little bug that is walking along the edge. From our point of view all we see is the point in front of us: a one-dimensional space.

<img align="center" width="300" src="https://bastian.rieck.me/images/manifolds_circle.svg" hspace="10">

[source](https://bastian.rieck.me/blog/2019/manifold/)

Now, why is this interesting? According to the **manifold hypothesis**, "high dimensional data tends to lie in the vicinity of a low dimensional manifold" (Fefferman et al., 2016). That mean that our machine learning models would be able to fit "simpler" subspaces instead of having to fit high-dimensional input space and dimension reduction is always a win.

### Denoising Autoencoder 

What does our denoising autoencoder (DAE) have to do with all of this? Let us remember how it looks like:

<img align="center" width="500" src="https://github.com/juelha/DenoisingAutoencoder/blob/main/doc/DAE.png" hspace="10">

Our DAE has to learn how to map corrupted inputs $\tilde{x}$ back to the original inputs $x$. For our circle example the red points are the noise.

<img align="center" width="500" src="https://github.com/juelha/DenoisingAutoencoder/blob/main/reports/circle/circle.png" hspace="10">


When putting in new inputs coordinates to the trained DAE, we see where the DAE maps them to -> closer to the manifold!

<img align="center" width="500" src="https://github.com/juelha/DenoisingAutoencoder/blob/main/reports/circle/circle_vectorfield.png" hspace="10">

That means our DAE not only approximates the function of a circle, but also of the vector field around it. 


## Run this project

### Option 1:
- Open [Google Colab](https://colab.research.google.com/notebook), click on the github tab and paste in the link ```https://github.com/juelha/DenoisingAutoencoder/blob/main/demo.ipynb```
- running the first line of the demo notebook will clone the repo into your google colab files

### Option 2: Run Locally 
- install the conda environment with ```conda env create -f env.yml```
- Clone the Repository or Download the ZIP File and extract.
- Open the demo.ipynb by using your preferred code editor or jupyter 



## Meme

<img align="center" width="500" src="https://github.com/juelha/DenoisingAutoencoder/blob/main/doc/spiderman_meme.png" hspace="10">

## Sources
Fefferman, C., Mitter, S., & Narayanan, H. (2016). Testing the manifold hypothesis. Journal of the American Mathematical Society, 29(4), 983-1049.  https://arxiv.org/pdf/1310.0425
