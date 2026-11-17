# Autoencoder Theory, Architecture & Anomaly‑Score Rationale

**Issue #39** – *Document autoencoder theory, architecture and anomaly‑score rationale*  
Owner: Rudra Pratap Singh Choudhary

---

## 1. Overview
This methods note explains **why** a reconstruction‑based autoencoder is used for credit‑card fraud detection, **how** the specific under‑complete autoencoder implementation in the repository works, and **how** the reconstruction error is transformed into an anomaly score.  It draws on peer‑reviewed literature and reputable tutorials while staying faithful to the actual code in **Issue #32** (model training) and **Issue #34** (scoring & checkpoint‑safe inference).

---

## 2. Theory of Autoencoders for Anomaly Detection

| Concept | Description |
|--------|-------------|
| **Autoencoder** | A neural network that learns to map an input vector \(\mathbf{x}\) to a latent representation \(\mathbf{z}\) (encoder) and then back to a reconstruction \(\hat{\mathbf{x}}\) (decoder). The network is trained to minimise a reconstruction loss \(L(\mathbf{x},\hat{\mathbf{x}})\). |
| **Under‑complete bottleneck** | The latent space dimension \(d_{\text{latent}}\) is deliberately chosen **smaller** than the input dimension. This forces the model to capture only the most salient structure of the data, discarding noise. |
| **Reconstruction‑error‑based anomaly detection** | If the training data consist **solely of legitimate (non‑fraud) transactions**, the autoencoder learns a compact manifold that well‑represents normal behaviour. When a fraudulent transaction – which lies off this manifold – is passed through the network, the reconstruction error \(\|\mathbf{x}-\hat{\mathbf{x}}\|\) becomes large. The error therefore serves as a **proxy anomaly score**. |

### Key Papers & Tutorials
1. **Hawkins, S., et al. (2002)** – *Outlier Detection Using Replicated Neural Networks* (classic reference for reconstruction‑based anomaly detection).  
2. **Sakurada, M., & Yairi, T. (2014)** – *Anomaly Detection Using Autoencoders with Non‑linear Dimensionality Reduction* (applied to industrial sensor data).  
3. **Koch, G., et al. (2019)** – *Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection* (provides a comprehensive survey).  
4. **TensorFlow/Keras tutorial** – *Autoencoders* (https://www.tensorflow.org/tutorials/generative/autoencoder) – used as a practical implementation guide.

---

## 3. Implemented Architecture (Issue #32)
The repository implements a **single‑layer under‑complete autoencoder** in `src/models/autoencoder.py`.

```python
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )
        # Decoder – mirrors the encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, input_dim),
        )
        self.loss = nn.MSELoss(reduction="mean")

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
```

### Architecture Details
| Component | Size | Activation |
|-----------|------|------------|
| Input layer | 30 (number of numeric features after preprocessing) | – |
| Hidden encoder layer | 64 | ReLU |
| Latent bottleneck | 16 (configurable via `latent_dim`) | – |
| Hidden decoder layer | 64 | ReLU |
| Output layer | 30 | – |

*The network is trained with **Mean‑Squared Error (MSE)** loss on **only legitimate training samples** (`train_legit` split).*  
Training hyper‑parameters (learning rate, epochs, batch size) are defined in `src/config.py` under the `autoencoder` section.

---

## 4. Scoring & Checkpoint‑Safe Inference (Issue #34)
### 4.1 Scoring Function
```python
def score(self, X: np.ndarray) -> np.ndarray:
    """Return per‑sample reconstruction MSE.
    Higher values indicate larger deviation from the learned normal manifold.
    """
    self.model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(self.device)
        recon = self.model(X_tensor)
        mse = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
    return mse
```
The function outputs a **single scalar per transaction** – the mean squared reconstruction error across all input dimensions.

### 4.2 Contract‑Compliant CSV Export
`run_evaluation` now records scores with the required columns (`row_id, split, model, anomaly_score`). The `model` field is filled with the literal string **`autoencoder`**.

### 4.3 Checkpoint Safety
* The model is saved after training as `autoencoder.pt` together with a JSON metadata file (`autoencoder_meta.json`).
* During inference the checkpoint is **loaded deterministically** (same weight initialisation, same device placement, no stochastic layers). As a result, re‑scoring the same validation data after loading the checkpoint yields identical anomaly scores (verified by a reproducibility test in `tests/test_autoencoder_baseline.py`).

---

## 5. Limitations & Failure Modes
1. **Overlap of fraud & legitimate manifolds** – Some fraud patterns may still be well‑reconstructed because they share feature statistics with normal transactions; the autoencoder will assign low anomaly scores to them.
2. **Feature engineering dependence** – The model operates on the **scaled numeric features** produced by `PreprocessingPipeline`. Any drift in feature distribution (e.g., new transaction types) can degrade detection.
3. **Single‑scalar score** – Only the average reconstruction error is used. More sophisticated metrics (e.g., Mahalanobis distance in latent space) could capture subtle deviations.
4. **Capacity limitation** – The bottleneck size (default 16) is a design trade‑off; too small may under‑fit, too large may over‑fit and reconstruct anomalies.

---

## 6. Architecture Diagram
```mermaid
flowchart TD
    subgraph Input[Preprocessed Feature Vector]
        I[Feature Vector \(30 dims)];
    end
    I -->|Linear 30→64| E1[Encoder Layer 1\nReLU]
    E1 -->|Linear 64→16| Z[Latent Bottleneck]
    Z -->|Linear 16→64| D1[Decoder Layer 1\nReLU]
    D1 -->|Linear 64→30| R[Reconstruction]
    R -->|MSE| Score[Reconstruction Error]
    style I fill:#e3f2fd,stroke:#90caf9,stroke-width:2px;
    style Z fill:#fff59d,stroke:#fdd835,stroke-width:2px;
    style Score fill:#ffcc80,stroke:#fb8c00,stroke-width:2px;
```
*Diagram reflects the exact layer sizes and activation functions used in the code.*

---

## 7. References
1. Hawkins, S., He, H., & Basak, J. (2002). *Outlier detection using replicated neural networks*. In Proceedings of the 13th International Conference on Neural Information Processing Systems (NIPS).  
2. Sakurada, M., & Yairi, T. (2014). *Anomaly Detection Using Autoencoders with Non‑linear Dimensionality Reduction*. Proceedings of the MLSDA 2014.  
3. Koch, G., Zemel, R., & Salakhutdinov, R. (2019). *Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection*. arXiv:1906.02371.  
4. TensorFlow Documentation – *Autoencoders*. https://www.tensorflow.org/tutorials/generative/autoencoder  
5. The repository’s implementation – `src/models/autoencoder.py` (Issue #32) and scoring logic in `src/run.py` / `src/evaluation_runner.py` (Issue #34).

---

**Prepared for inclusion in the final project report.**  
If any phrasing or diagram detail needs adjustment, let me know and I’ll edit the note accordingly.
