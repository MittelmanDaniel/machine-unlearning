import streamlit as st
from PIL import Image

st.set_page_config(page_title="Midterm", page_icon="📃")

st.title("Midterm")


st.header("Introduction/Problem Definition")


st.write(
    """
    As machine learning (ML) becomes more prevalent in our society, there is a growing concern about the
    sensitive information used to train some of these models.  While users can choose to delete their data from certain
    websites, this is not so easy for ML models.  The recent field of **machine unlearning** aims to allow models to
    "forget" certain training data, while still retaining the knowledge gained from the rest of the data.  Not only does
    this address privacy concerns, but it also protects against adversarial attacks, biased data, manipulated data, and outdated data. [1].
    
    A naive approach to machine unlearning would be simply retraining the model on the "retain set", or everything 
    except the data to be forgotten, the "forget set".  However, this is computationally expensive or impossible if the 
    full training data is not available.  
    
    Existing methods can be characterized into model-agnostic, model-specific, and data-driven approaches. Recent model-
    agnostic approaches include Stochastic Teacher Networks, Selective Synaptic Dampening, and EU-k/CF-k [4][5][6]. 
    EU-k and CF-k, short for Exact-Unlearning Last k Layers and Catastrophically Forgetting the Last k Layers, are 
    baselines that achieve inexact unlearning by freezing upstream layers and only retraining lower layers [6]. 
    Model-specific methods target specific architectures like Graph Neural Networks. Finally, data-driven approaches 
    use data augmentation, splitting, and utilize specific metrics to speed up retraining.

    There's some standard terminology we will use. The forget set is the set of data we want to unlearn, the retain set is the set of training data
    we want to keep, and the test set is the standard definition.

    """)

st.subheader("Problem Definition Specifics")

st.write("""
    In this project, we will tackle the problem of forgetting face images from an age predictor, as outlined in the
    NeurIPS 2023 Machine Unlearning Challenge [2]. The goal will be to modify the trained model such that it is still accurate
    and similar to if the model were trained without the images we want to forget. We implement some model-agnostic methods proposed in recent 
    literature and will develop a novel unlearning algorithm.  Since the data is hidden for the challenge, we will evaluate 
    our methods on the simplified UTK dataset [3].
    """
)

st.header("Methods")

st.write(
    """
    We implement Selective Synaptic Dampening, Optimizing Last-K Layers, and will implement Catastrophic forgetting.  
    Then, we aim to develop an unlearning algorithm inspired from existing techniques.  We used
    the NeurIPS challenge kit [7] as a starting point.
    """
)

st.subheader("Data Preprocessing")
st.write(
    """
    The dataset contains 23705 rows and 5 columns. Each row is a data point,
    where column 1 = age, column 2 = ethnicity, column 3 = gender, column 4 = image name, and column 5 = pixels. 
    We want to go from an image to an age, so pixels will be our input and our labels will be the age.
    
    We normalized the pixel values from 0-255 to 0-1.0 in order to keep training stable.
    
    Resnet expects 3 channels, so we duplicated the 1 channel image across 3 channels. This is okay because a black and white image
    is equivalent to RGB with the same values in all 3 channels.
    
    We didn't use image augmentation when training because that would make us need to unlearn the augmentations 
    as well which is much more challenging.
    """
)

st.image(Image.open('./app/images/dataset_random_images.png'), caption="Example Images")

st.subheader("Algorithm/Model")

st.write("""
    In order to unlearn data from an age prediction model, we must first train an age prediction model. We use a Resnet-18 architecture
    that was pretrained on ImageNet, we then freeze the first 5 residual blocks,
        and then continue training on the rest of the layers using our dataset. We replace the last layer with an age regression instead of classification.
        We split our data into 90% training and 10% testing and fine-tuned for 30 epochs.
        Batch size of 256, learning rate of 0.001.
    
    We used mean squared error as the loss function, which makes sense as this is a regression problem.


    The first method we implemented is called Exact-unlearning the last k layers (EU-k) [6]. Only the last k layers are retrained
    on the retained dataset. The rest of the layers are untouched, or frozen.
    
    The second method we intend to implement is called Catastrophically forgetting the last k layers (CF-k) [6]. In this method, the last k
    layers are completely reinitialized and then trained on the retained dataset, while the rest of the layers are also frozen.
    
    We then implemented a retraining-free method called Selective Synaptic Dampening (SSD) [7].
    SSD uses the Fisher Information Matrix (FIM) to determine the importance of the weights.
    If a weight is far more important for the forget set than the whole set, then we dampen the weight.
    This results in the model forgetting the data in the forget set while minimizing the impact on the retain set.
    More formally, for each weight $\\theta$, we dampen the weight as follows:
    """
)

st.latex(r'''
    \begin{align*}
        \beta &= \min\left(\frac{\lambda\,\text{FIM}_{\text{all}}(\theta)}{\text{FIM}_{\text{forget}}(\theta)}\theta, 1\right) \\
        \theta &=
        \begin{cases}
        \beta \theta & \text{if } \text{FIM}_{\text{forget}}(\theta) > \text{FIM}_{\text{all}}(\theta) \\
        \theta & \text{otherwise}
        \end{cases}
    \end{align*}
    ''')

st.markdown("""
    where $\\lambda$ and $\\alpha$ are hyperparameters that are the threshold factor for selecting which weights to dampen and the dampening factor, respectively.
    We used $\\lambda = 50$ and $\\alpha = 0.1$.
    """
)

st.header("Results")

st.write(
    """
    While we could completely retrain the model on the leftover data, this is expensive.
    Our goal is to make an algorithm that will run in a fraction of the time it takes to retrain and is most similar to retraining the model.
    
    There are a few things to consider:
    - **Accuracy.**  We will evaluate the accuracy of an unlearned model on the retain set and a test set and try to keep them as good or better than retraining.
    - **Forget Quality.**  Measuring this is not standardized, but below are some potential metrics:  
        1. Failure rate of measurement inference attacks, which attempt to infer if a model was trained on a piece of data.  If we successfully mimic a retrained model, these attacks should fail [8].
        2. KL divergence of the probability distribution our unlearned model predicts and the distribution that the retrained model predicts [9].
        3. Google's method involving delta-epsilon differential privacy.  The math is too involved to explain here within the word limit—please see [8].
    """
)


st.write("This is the loss while doing transfer learning on our dataset")
st.image(Image.open('./app/images/training_history_losses.png'))

st.write(
    """
         The validation loss here is horrible and implies overfitting.
         However our goal is unlearning and so this exaggerates the difference between the train set and test set,
         making it so our model is exceptionally vulnerable to membership inference attacks and so there's a greater signal
         when our unlearning algorithms run because the difference between trained on images and not trained on images is greater.
    """
)

st.image(Image.open('./app/images/dataset_predicted_images.png'), caption="Example Age Predictions On Test Set")

st.write("You can see the distribution of ages for the predictions and true are similar so our algorithm is understanding the distribution")

st.image(Image.open('./app/images/distribution_prediction_training.png'))

st.image(Image.open('./app/images/distribution_prediction_test.png'))



st.image(Image.open('./app/images/distribution_labels_actual.png'))


st.write("You can see the losses for the train set are lower than the test set ideally an unlearning algorithm will make the forget set's similar to the test set. We had difficulties with that plot after unlearning so it is not ready at the moment")

st.image(Image.open('./app/images/losses_train_v_test.png'))

st.write("Now we discuss results for our unlearning baseline")

st.write("Before unlearning the Membership Inference Attack had a 61% success rate, after exact k unlearning with k=5 for 30 epochs it dropped to 57%. This is a sign the algorithm worked somewhat but we can do better. A larger k would allow more of the model to fit the retain set, but too much like k = 20 make MIA's succeed 78% of the time due to overfitting so hard on the retain set that it makes it more obvious")

st.subheader("Next Steps")

st.write(
    """
    Now that we have implemented some baseline models for unlearning, we hope to finish the rest of the baselines and explore novel approaches to further reduce MIA success and increase accuracy.
    """
)

st.header("References")

st.write(
    """
[1] T. T. Nguyen, T. T. Huynh, P. L. Nguyen, A. W. Liew, H. Yin, Q. V. H. Nguyen, “A Survey of Machine Unlearning,” ACM, Oct. 2022, https://doi.org/10.1145/nnnnnnn.nnnnnnn

[2] “NeurIPS 2023 Machine Unlearning Challenge.” https://unlearning-challenge.github.io (accessed Oct. 6, 2023).

[3] "AGE, GENDER AND ETHNICITY (FACE DATA) CSV" https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv (accessed Nov. 13, 2023).

[4] X. Zhang, J. Wang, N. Cheng, Y. Sun, C. Zhang, J. Xiao, “Machine Unlearning Methodology base on Stochastic Teacher Network,” 2023, arXiv:2308.14322v1

[5] J. Foster, S. Schöpf, A. Brintrup, “Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening,” 2023, arXiv:2308.07707v1

[6] S. Goel, A. Prabhu, A. Sanyal, S. Lim, P. Torr, P. Kumaraguru, “Towards Adversarial Evaluations for Inexact Machine Unlearning,” 2023, arXiv:2201.06640v3 

[7] “Unlearning Challenge Starting Kit.” https://github.com/unlearning-challenge/starting-kit (accessed Oct. 6, 2023).

[8] “NeurIPS 2023 - Machine Unlearning.” https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/data (accessed Oct. 6, 2023).

[9] S. Mercuri, R. Khraishi, R. Okhrati, D. Batra, C. Hamill, T. Ghasempour, A. Nowlan, “An Introduction to Machine Unlearning,” 2022, arXiv:2209.00939v1 
    """
)

st.header("Appendix")

st.subheader("Timeline")
st.write("https://docs.google.com/spreadsheets/d/12QYt-xssTOid3kKTiW-Wiqj-zZz-9jf8yIOI0-dXKJE/edit?usp=sharing")

st.subheader("Contributions")
st.write(
    """
    Adrian Cheung
     - Selective Synaptic Dampening attempted implementation
     - Identifying better dataset
    
    Henry Zhang
     - Transfer learning Resnet-18 on Age Regression
     - Membership Inference Attack code

    Daniel Mittelman
     -  Implemented Last K layer Unlearning
     -  Implemented Loss Distribution calculation
     -  Modify Proposal to fit Midterm report
     

    Yutao Xu
     - Modify Proposal to fit Midterm Report
     - Formatted images correctly in proposal.
        """
)