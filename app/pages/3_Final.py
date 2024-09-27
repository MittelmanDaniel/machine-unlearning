import streamlit as st

st.set_page_config(page_title="Final", page_icon="📃")

st.title("Final")


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
    full training data is not available. Given a trained model, we aim to approximate as if it were trained without a certain portion of their data.
    
    Existing methods can be characterized into model-agnostic, model-specific, and data-driven approaches. Recent model-
    agnostic approaches include Stochastic Teacher Networks, Selective Synaptic Dampening, and EU-k/CF-k [4][5][6]. 
    EU-k and CF-k, short for Exact-Unlearning Last k Layers and Catastrophically Forgetting the Last k Layers, are 
    baselines that achieve inexact unlearning by freezing upstream layers and only retraining lower layers [6]. 
    Model-specific methods target specific architectures like Graph Neural Networks. Finally, data-driven approaches 
    use data augmentation, splitting, and utilize specific metrics to speed up retraining.
    """)

st.subheader("Problem Definition Specifics")

st.write("""
    In this project, we tackle the problem of forgetting face images from an age predictor, as outlined in the
    NeurIPS 2023 Machine Unlearning Challenge [2]. The goal is to modify the trained model such that it is still accurate
    and similar to if the model were trained without the images we want to forget. We implemented some model-agnostic methods proposed in recent 
    literature. Since the data is hidden for the challenge, we evaluated our methods on the simplified UTK dataset [3]. 
    This dataset contains black and white images of faces labelled with their ages
    """
)

st.header("Methods")

st.write(
    """
    We implement Selective Synaptic Dampening, Optimizing Last-K Layers, and Forgetting last k layers.  
    We used the NeurIPS challenge kit [7] as a starting point.
    We used a Resnet-18 architecture to predict ages from an image so we can then apply our unlearning algorithms to it
    """
)

st.subheader("Data Preprocessing")
st.write(
    """
    The dataset contains 23705 rows and 5 columns. Each row is a data point,
    where column 1 = age, column 2 = ethnicity, column 3 = gender, column 4 = image name, and column 5 = pixels. 
    We want to go from an image to an age, so pixels will be our input and our labels will be the age.
    >>
    We normalized the pixel values from 0-255 to 0-1.0 in order to keep training stable.
    
    Resnet expects 3 channels, so we duplicated the 1 channel image across 3 channels. This is okay because a black and white image
    is equivalent to RGB with the same values in all 3 channels.
    
    We didn't use image augmentation when training because that would make us need to unlearn the augmentations 
    as well which is much more challenging.
    """
)

st.image("./app/images/dataset_random_images.png", caption="Example Images")

st.subheader("Algorithm/Model")

st.subheader("Age Regression")
st.write("""
    In order to unlearn data from an age prediction model, we must first train an age prediction model. We use a Resnet-18 architecture
    that was pretrained on ImageNet, we then freeze the first 5 residual blocks,
        and then continue training on the rest of the layers using our dataset. We replace the last layer with a single output for regression instead of classification.
        We split our data into 90% training and 10% testing and fine-tuned for 30 epochs.
        Batch size of 256, learning rate of 0.001.
    
    We used mean squared error as the loss function, which makes sense as this is a regression problem.
         
    We chose this architecture because it's good at image based tasks and we could finetune a pretrained one.
""")

st.subheader("Playing with the Last K")
st.write("""

    The first method we implemented is called Exact-unlearning the last k layers (EU-k) [6]. The last k layers are completely 
    reinitialized and then retrained
    on the retained dataset. The rest of the layers are untouched, or frozen.
         
    Another method we implemented is calle Catastrophic-Forgetting (CF-k) which is fine-tuning the last k layers [6]. 
    The weights are not reinitialized but we still fine-tune the last k layers.
         
    These methods are similar to the ideal of completely retraining the network and finetuning the network on just the retain set, but require less compute because the number of layers being trained is smaller.
         

""")

st.subheader("Selective Synaptic Dampening")

st.write("""
    We then implemented a retraining-free method called Selective Synaptic Dampening (SSD) [7].
    SSD uses the Fisher Information Matrix $F$ to determine the importance of the weights for different sets.
    If a weight is far more important for the forget set than the whole set, then we dampen the weight.
    This results in the model forgetting the data in the forget set while minimizing the impact on the retain set.
    More formally, for each weight $\\theta$, we dampen the weight as follows:
    """
)

st.latex(r'''
    \begin{align*}
        \beta &= \min\left(\frac{\lambda\,F_{\text{all}}(\theta)}{F_{\text{forget}}(\theta)}\theta, 1\right) \\
        \theta &=
        \begin{cases}
        \beta \theta & \text{if } F_{\text{forget}}(\theta) > \alpha F_{\text{all}}(\theta) \\
        \theta & \text{otherwise}
        \end{cases}
    \end{align*}
    ''')

st.markdown("""
    where $\\alpha$ and $\\lambda$ are hyperparameters that are the threshold factor for selecting which weights to dampen and the dampening factor, respectively.
    """
)

#     0: "0-6 years old",
#     1: "7-12 years old",
#     2: "13-19 years old",
#     3: "20-30 years old",
#     4: "31-45 years old",
#     5: "46-55 years old",
#     6: "56-66 years old",
#     7: "67-80 years old",
#     8: "81+ years old",

st.subheader("Age Classification")
st.markdown("""
    However, SSD was originally designed for classification tasks, which is also used in the NeurIPS challenge.
    In addition to regression, we developed an age classifier by splitting the ages into 9 bins: 0-6, 7-12, 13-19, 20-30, 31-45, 46-55, 56-66, 67-80, and 81+.
    
    Similar to age regression, we started with a Resnet-18 architecture that was pretrained on ImageNet.
    To prevent overfitting, we replaced the last 2 sets of convolutional layers a simple convolutional layer:
    """
)

st.image("./app/images/resnet_changes.png", caption="Age Classification:  Modified ResNet-18")

st.markdown("""
    We trained the model for 30 epochs with weighted cross entropy loss to handle class imbalance.
    We used SGD as the optimizer with a learning rate of 0.01 and momentum of 0.9, with an exponential learning rate scheduler that decays the learning rate by 0.5 every 5 epochs.
    
    With classification, we tested SSD on two tasks:  **class unlearning** and **random unlearning**.
    Class unlearning is when we unlearn all data from a single class, and random unlearning is when we unlearn random data points.
    The latter was the original goal and is more challenging because it's harder to unlearn weights that are tied to specific pieces of data without sacrificing accuracy on the retain set.
    """
)

st.subheader("SSD-FT:  Combining SSD and Fine-Tuning")
st.markdown("""
    We introduce a hybrid approach called **SSD-FT** that uses both SSD and fine-tuning with layer freezing.
    In the simplest case, we first use SSD to unlearn the forget set, and then fine-tune on the retain set to 
    relearn any mistakenly unlearned weights while catastrophically forgetting the forget set.
    Note that this is not a model layer but a step in an algorithm.
    """
)

st.image("./app/images/df_block.png", caption="SSD-FT:  Dampening-Finetuning Block")

st.markdown("""
    We call this a dampening-finetuning block, or **DF block**.
    If the fine-tune step freezes the first $k$ layers, no dampening should be applied on the first $k$ layers.
    
    For more complex unlearning tasks, we can chain multiple DF blocks.
    Every subsequent block should have less aggressive dampening and less specific fine-tuning that relearns more of the later layers.
    In other words, there is generally an increase in dampening parameters $\\alpha$, $\\lambda$ and number of frozen layers $k$.
    """
)

st.image("./app/images/chaining_df_blocks.png", caption="SSD-FT:  Chaining DF Blocks")


st.header("Results")

st.write(
    """
    While we could completely retrain the model on the leftover data, this is expensive.
    Our goal is to make an algorithm that will run in a fraction of the time it takes to retrain and is most similar to retraining the model.
    
    We use these metrics to determine the success of our unlearning algorithms:
    - **Accuracy/Loss.**  We evaluate the accuracy/loss of an unlearned model on the retain set and a test set and try to keep them as good or better than retraining.
    - **Forget Quality.**  Failure rate of membership inference attacks, which attempt to infer if a model was trained on a piece of data.  If we successfully mimic a retrained model, these attacks should fail [8].
    """
)

st.subheader("Membership Inference Attack")

st.write("""To evaluate the success of our unlearning algorithms we train a logistic regression that will predict if something is in
         the forget set or the train set given the loss. This way if the loss distributions are further apart the logistic regression will fail.
         This setup is kind of like a GAN but we don't use the logistic regression in training, we use it to test our algorithms. """)


st.subheader("Age Regression")

st.write("Below is the loss while doing transfer learning on our dataset:")
st.image("./app/images/training_history_losses.png")

st.write(
    """
         The validation loss here is horrible and implies overfitting.
         However our goal is unlearning and so this exaggerates the difference between the train set and test set,
         making it so our model is exceptionally vulnerable to membership inference attacks and so there's a greater signal
         when our unlearning algorithms run because the difference between trained on images and not trained on images is greater.
    """
)

st.image("./app/images/dataset_predicted_images.png", caption="Example Age Predictions On Test Set")

st.write("You can see the distribution of ages for the predictions and true are similar so our algorithm is understanding the distribution:")

st.image([
    "./app/images/distribution_prediction_training.png",
    "./app/images/distribution_prediction_test.png",
    "./app/images/distribution_labels_actual.png",
])


st.write("""Below is the loss distribution for the forget set and the test set. Notice how they are very different
         so attacks are very successful because given a tiny loss it's more likely it was in the forget set.
         Ideally our algorithms will make the forget set's loss look less like the train set and more like the test set """)

st.image("./app/images/no_forgetting.png", caption="No Forgetting")

st.subheader("Baseline:  Training On Retain Set")
st.write(""" 
    Below is the loss for every epoch when training the model on just the retain set. 
    This is the ideal baseline we try to approximate. Normally it's impractical due to large training times, but in our toy example we can do it.
    Note how the forget set loss is identical to the validation set loss because the model has never seen it
""")
st.image("./app/images/baseline.png", caption="Baseline (Training On Just Retain Set)")

st.write(""" Below is the distribution of the losses for the forget set and test set.
         Notice how they are very similar and so the MIA attack is only right about half the time which is equivalent to random guessing.
         This is the ideal unlearning algorithm and what we will try to approximate.""")

st.image("./app/images/baseline_2.png", caption="Baseline (Training On Just Retain Set)")

st.subheader("Finetuning the last k layers")

st.write("""
    Membership Inference Attack went from 74% to 69.8% success rate after finetuning k unlearning, with k = 5, epochs = 90, learning rate = 0.001.
    Result shows signs of forgetting. Larger k would allow more of model to fit on retain set. However, very large k values would take more compute and potentially overfit the retain set.
    """
)

st.write("#### CF-k (finetuning no reset) ####")

st.image("./app/images/cf-k.png", caption="CF-k (finetuning no reset)")

st.write("Finetuning on the retain set after 30 epochs. Attack accuracy increases to 0.813, which is more than the attack accuracy of the original model (0.740). This is because although the loss distribution is spreading out, the test distribution spread out faster so there's more test set batches that are further from the forget set.")

st.image("./app/images/cf-k_2.png", caption="CF-k (finetuning no reset)")

st.write("Finetuning on the retain set after 90 epochs. Attack accuracy finally decreases to 0.698, less than the attack accuracy of the original model")


st.subheader("EU-k")

st.write("""Before unlearning the Membership Inference Attack had a 74% success rate, 
         after exact k unlearning with k=5 for 30 epochs it dropped to 66.4%. 
         This is significantly better than CF-k which makes sense because we're completely reinitializing.
         A larger k would allow more of the model to fit the retain set, but increases computational cost""")

st.image("./app/images/eu-k.png", caption="EU-k (finetuning, reinitializing weights)")

st.image("./app/images/regression_results.png", caption="Table of unlearning results for age regression")

st.subheader("Selective Synaptic Dampening")

st.markdown(
    """
    For the regression task, SSD resulted in very little change to the model weights and thus did not unlearn the data.
    We believe this is because SSD was designed for classification problems, so we will focus on the classification tasks.

    #### Class Unlearning
    Class unlearning proved to be an easy task for SSD.
    Suppose we wanted to wipe all young children off the world—for AI safety reasons, of course.
    We can use SSD to unlearn the 0 class, or 0-6 year olds:
    """
)

st.image("./app/images/class_unlearning_age_0-6.png", caption="Loss/Accuracy for SSD Class Unlearning on Age 0-6")

st.markdown(
    """
    You can see the retain loss/accuracy is unaffected, but the forget loss is huge and accuracy goes to 0.
    This is the ideal scenario for unlearning.
    The confusion matrices below also show that the model is unlearning 0-6 year olds and guessing the most similar class, 7-12 year olds.
    """
)

st.image([
    "./app/images/train_before_SSD.png",
    "./app/images/train_after_SSD_class_0.png",
    "./app/images/test_before_SSD.png",
    "./app/images/test_after_SSD_class_0.png",
])

st.markdown("#### Random Unlearning")

st.image("./app/images/random_unlearning.png", caption="Loss/Accuracy for SSD Random Unlearning")

st.markdown("""
    SSD had little effect on the model weights in random unlearning.
    With more aggressive dampening, SSD unlearned the forget set but also unlearned the retain set.
    This motivated us to try the hybrid SSD-FT approach.
    """
)

st.subheader("SSD-FT")

st.image("./app/images/SSD_FT_random_unlearning.png", caption="Loss/Accuracy for SSD-FT Random Unlearning")

st.markdown("""
    We tested many different configurations of SSD-FT, and above we show the best for each number of SSD-FT blocks.
    We also trained a baseline model on just the retain set to show the ideal unlearning.
    We found that 3 SSD-FT blocks was the best, as proven by the very similar loss/accuracy to the baseline.
    Additional DF blocks either "overfit" or forget too much.
    """
)

st.image("./app/images/SSD_FT_random_unlearning_2.png", caption="MIA Scores for SSD-FT Random Unlearning", width=300)

st.markdown("""
    Looking at the MIA success rates, we see that SSD-FT with 3 DF blocks is able to get the MIA success rate down to 
    nearly 50%, the perfect MIA score because the model is guessing randomly.
    

    We believe that SSD-FT had such an improvement over SSD because SSD-FT can afford to more aggressively dampen with
    fine-tuning directly following it.  Sometimes it is necessary to "reset" some of the weights important to both the 
    forget and retain set, to learn weights more specific to the retain set—this is impossible with SSD alone.
    SSD-FT combines the speed and forgetting potential of SSD and the retain potential of fine-tuning.
    """
)

st.header("Conclusion")

st.write("""None of the unlearning methods on their own were able to unlearn on their own. 
         The last-k methods retained accuracy but didn’t do much for MIA attacks. 
         SSD does well on class unlearning, but poorly when the forget set is uniformly taken from classes.
         SSD-FT is rather fast and produces good unlearning results. We hope to experiment with DF blocks and similar techniques.""")

st.header("Next Steps")

st.write(
    """
    We have investigated SSD and last-k unlearning methods and combined them to form a novel unlearning method with modest success.
    Future work could investigate the effectiveness of these methods on other architectures and investigate the 
    difference in effectiveness in classification and regression tasks.
    """
)

st.header("References")

st.write(
    """
    [1] T. T. Nguyen, T. T. Huynh, P. L. Nguyen, A. W. Liew, H. Yin, Q. V. H. Nguyen, “A Survey of Machine Unlearning,” ACM, Oct. 2022, https://doi.org/10.48550/arXiv.2209.02299

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
     - Modified ResNet age classification model
     - Selective Synaptic Dampening
     - SSD-FT
    
    Henry Zhang
     - Transfer learning Resnet-18 on Age Regression
     - Generated Figures and Tables for other unlearning algos, CNN training
     - Membership Inference Attack code

    Daniel Mittelman
     -  Implemented Last K layer Unlearning
     -  Implemented Loss Distribution calculation
    
    Yutao Xu
     - Writing report/slides
     - Active Member, participated in meetings
    """
)