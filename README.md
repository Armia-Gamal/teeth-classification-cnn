All work done in Kaggle environment.

## Dataset & Preprocessing
- Images resized and normalized to [0,1]  
- Augmentation: rotation, horizontal flip, zoom  
→ reduces overfitting, improves generalization

## Data Visualization
- Class distribution plots  
- Original vs augmented image pairs  
- Accuracy & loss curves (train vs validation)

## Model Architecture
Custom CNN with:  
Conv2D → MaxPooling2D → GlobalAveragePooling2D → Dense → Dropout

## Training Results
- Strong baseline performance  
- High train/val accuracy  
- Smooth loss convergence  
- No major overfitting  
- Minor val fluctuations due to augmentation

## Saved Model
model/teeth_classifier_model.h5

Load example:
```python
from tensorflow.keras.models import load_model
model = load_model("model/teeth_classifier_model.h5")
