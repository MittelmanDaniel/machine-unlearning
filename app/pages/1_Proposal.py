import streamlit as st

st.set_page_config(page_title="Proposal", page_icon="📃")

st.title("Proposal")

st.header("Problem & Background")

st.write(
    """
    As machine learning (ML) becomes more prevalent in our society, there is a growing concern about the
    sensitive information used to train some of these models.  While users can choose to delete their data from certain
    websites, this is not so easy for ML models.  The recent field of **machine unlearning** aims to allow models to
    "forget" certain training data, while still retaining the knowledge gained from the rest of the data.  Not only does
    this address privacy concerns, but it also protects against adversarial attacks and biased data [1].
    
    A naive approach to machine unlearning would be simply retraining the model on the "retain set", or everything 
    except the data to be forgotten, the "forget set".  However, this is computationally expensive or impossible if the 
    full training data is not available.  
    
    Existing methods can be characterized into model-agnostic, model-specific, and data-driven approaches. Recent model-
    agnostic approaches include Stochastic Teacher Networks, Selective Synaptic Dampening, and EU-k/CF-k [4][5][6]. 
    EU-k and CF-k, short for Exact-Unlearning Last k Layers and Catastrophically Forgetting the Last k Layers, are 
    baselines that achieve inexact unlearning by freezing upstream layers and only retraining lower layers [6]. 
    Model-specific methods target specific architectures like Graph Neural Networks. Finally, data-driven approaches 
    use data augmentation, splitting, and utilize specific metrics to speed up retraining.
    
    In this project, we will tackle the problem of forgetting face images from an age predictor, as outlined in the
    NeurIPS 2023 Machine Unlearning Challenge [2].  We hope to implement some model-agnostic methods proposed in recent 
    literature and develop a novel unlearning algorithm.  Since the data is hidden for the challenge, we will evaluate 
    our methods on the CelebA dataset, which contains over 200,000 celebrity face images with with age [3].
    """
)

st.header("Methods")

st.write(
    """
    We will implement some of the aforementioned methods that can be applied to the pretrained age
    predictor.  Then, we aim to develop an unlearning algorithm inspired from existing techniques.  We will use 
    PyTorch and the NeurIPS challenge kit [7] as a starting point.
    """
)

st.header("Potential Results")

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

st.header("References")

st.write(
    """
[1] T. T. Nguyen, T. T. Huynh, P. L. Nguyen, A. W. Liew, H. Yin, Q. V. H. Nguyen, “A Survey of Machine Unlearning,” ACM, Oct. 2022, https://doi.org/10.1145/nnnnnnn.nnnnnnn

[2] “NeurIPS 2023 Machine Unlearning Challenge.” https://unlearning-challenge.github.io (accessed Oct. 6, 2023).

[3] Z. Liu, P. Luo, X. Wang, X Tang, “Large-scale CelebFaces Attributes (CelebA) Dataset.” https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (accessed Oct. 6, 2023).

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
     - Github, Streamlit setup
     - Background
     - Problem Definition
     - Methods
     - Gantt Chart
    
    Henry Zhang
     - Wrote existing methods paragraph
     - Contributed ideas for potential novel approaches
     - Gantt Chart
     - Made presentation slides
     - Voiceover for presentation

    Daniel Mittelman
     - Potential Results and Discussion
     - Contributed ideas for potential novel approaches
     - Discovered kaggle competition
     - Made presentation slides
     - Voiceover for presentation

    Yutao Xu
     - Formatted Citations in IEEE format
    
    Ben Jiras
    """
)

st.subheader("Video Presentation")
st.write("https://drive.google.com/file/d/1oHCOv8PsF5SrbTesH2qOsRIWBagCRNp9/view?usp=sharing")
