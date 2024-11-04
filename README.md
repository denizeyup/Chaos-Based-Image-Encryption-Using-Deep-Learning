# Chaos-Based-Image-Encryption-Using-Deep-Learning

## Project Overview

This project is part of an undergraduate thesis conducted at Gazi University, which explores a novel approach to image encryption using chaotic systems and deep learning models. The main focus is on securing grayscale brain tomography images by encrypting them through the combination of chaotic systems and Generative Adversarial Networks (GANs). The project ensures that the images can be transmitted securely over untrusted networks.

## Key Highlights

- **Generative Adversarial Network (GAN)**: The project uses a GAN model comprising a Generator and a Discriminator. The Generator encrypts the input images, while the Discriminator distinguishes between real and fake (encrypted) images.
- **Chaotic System**: The Logistic Map is utilized to generate chaotic sequences that are crucial for the encryption process.
- **Loss Functions**:
  - **Mean Squared Error (MSE)** is used to ensure minimal difference between the original and the generated images.
  - **Structural Similarity Index (SSIM)** is employed to maintain the structural similarity between the images, preserving important details during encryption.
- **Noise Injection**: Gaussian noise is added to the input images to simulate real-world scenarios and improve the robustness of the encryption.

## Project Structure

- The project is structured into several key scripts and directories:
  - **Model Implementation**: This includes the definition of the GAN model with the Generator and Discriminator components.
  - **Utility Functions**: Various helper functions are provided for tasks like image preprocessing, noise addition, and evaluation metrics.
  - **Training and Testing Scripts**: Separate scripts are available to train the model on a dataset of grayscale images and to test the trained model on new data.
  - **Data Directory**: This is where the image dataset is stored, consisting of grayscale images used for training and testing the model.

## Requirements

The project requires Python 3.x along with several key libraries:
- TensorFlow for building and training the deep learning models.
- NumPy for numerical operations and handling image data.
- OpenCV for image processing tasks.
- Matplotlib for visualizing the images and results.

## Installation

To set up the environment for this project, you need to install the required libraries. The installation process involves using pip to install TensorFlow, NumPy, OpenCV, and Matplotlib. Detailed installation instructions are provided to ensure a smooth setup.

## Usage

### Training the Model

To train the GAN model:
1. Clone the repository to your local machine.
2. Place your grayscale image dataset in the designated `data/` directory.
3. Run the training script provided in the repository. The script will process the images, add noise, and train the GAN model on this dataset.

### Testing the Model

Once the model is trained, you can test it using the testing script. This script will take a new set of images, apply the same noise and chaotic encryption process, and visualize the results. The performance of the model is assessed based on its ability to reconstruct the original images from the encrypted ones.

## Model Architecture

The GAN model consists of two main components:
- **Generator**: Responsible for encrypting the input images. It uses convolutional layers with residual blocks to generate high-quality encrypted images.
- **Discriminator**: Evaluates the authenticity of the encrypted images by distinguishing between real and generated images. It is trained to improve the Generator's performance by providing feedback on the generated images.

The entire model is optimized using Adam optimizer, and the training process alternates between updating the Generator and the Discriminator to achieve a balance where the Generator produces convincing encrypted images.

## Results

The project demonstrates effective image encryption and decryption. The use of chaotic systems in combination with deep learning ensures robust encryption that resists various attacks. The performance metrics, including MSE and SSIM, indicate high fidelity and structural similarity between the original and reconstructed images, validating the effectiveness of the proposed approach.

## References

This project is based on the thesis titled *"Chaotic Based Image Encryption Using Deep Learning"* by Eyüp Deniz, under the supervision of Assoc. Prof. Dr. Hüseyin Polat at Gazi University.

## License

This project is licensed under the MIT License, allowing for open use and modification of the codebase. For more details, refer to the LICENSE file in the repository.

## Acknowledgements

- **Eyüp Deniz**: Project developer and thesis author.
- **Assoc. Prof. Dr. Hüseyin Polat**: Project supervisor and advisor.
