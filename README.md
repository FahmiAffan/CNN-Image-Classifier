# CNN Image Classifier - CIFAR-10

A modern Flask web application for image classification using a trained CIFAR-10 Convolutional Neural Network model.

## Features

- **Real CIFAR-10 Model**: Uses a trained CNN model for accurate image classification
- **OpenCV Integration**: Optimized image processing with OpenCV for better performance
- **10 Object Classes**: Classifies images into airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck
- **Image Preview**: Real-time preview of uploaded images
- **Confidence Scores**: Shows prediction confidence with visual progress bars

## Quickstart

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
python app.py
```

4. Open in browser: `http://localhost:5000`

## Project Structure

```
CNN-Image-Classifier/
├── app.py                 # Main application entry point
├── app/
│   ├── __init__.py       # Flask app factory
│   ├── model.py          # CIFAR-10 model implementation
│   └── routes.py         # Flask routes and handlers
├── models/
│   └── model_cifar10.h5  # Trained CIFAR-10 model
├── templates/
│   ├── index.html        # Upload page with modern UI
│   └── result.html       # Results page with predictions
├── static/
│   └── css/
│       └── style.css     # Custom styles (legacy)
├── uploads/              # Temporary file storage
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Dependencies

- **Flask 3.0.3**: Web framework
- **TensorFlow 2.15.0**: Deep learning framework for model loading
- **OpenCV 4.8.1.78**: Computer vision library for image processing
- **NumPy 2.1.2**: Numerical computing
- **Pillow 10.4.0**: Image processing
- **Werkzeug 3.0.3**: WSGI utilities

## Model Information

The application uses a pre-trained CIFAR-10 CNN model that can classify images into 10 categories:

- **Airplane**: Aircraft and flying vehicles
- **Automobile**: Cars and road vehicles
- **Bird**: Various bird species
- **Cat**: Domestic and wild cats
- **Deer**: Deer and similar animals
- **Dog**: Dogs and canines
- **Frog**: Frogs and amphibians
- **Horse**: Horses and equines
- **Ship**: Boats and watercraft
- **Truck**: Trucks and large vehicles

## Technical Details

### Image Processing Pipeline

1. **Upload**: User uploads image through web interface
2. **Preprocessing**: OpenCV processes image (resize to 32x32, normalize to [0,1])
3. **Prediction**: CIFAR-10 model predicts class and confidence
4. **Results**: Beautiful results page displays prediction with confidence score

### Performance Optimizations

- **OpenCV Processing**: Faster image operations compared to PIL
- **Model Caching**: Model loaded once and cached for subsequent requests
- **Stream Processing**: Direct file stream processing without temporary files
- **Responsive Design**: Optimized for all device sizes

## Usage

1. **Upload Image**: Click the upload area or drag & drop an image
2. **Preview**: See a preview of your uploaded image
3. **Predict**: Click "Prediksi Gambar" to classify the image
4. **View Results**: See the prediction with confidence score and class information

## Configuration

- **Max Upload Size**: 10 MB (configurable in `app/__init__.py`)
- **Supported Formats**: PNG, JPG, JPEG
- **Model Path**: `models/model_cifar10.h5`
- **Image Size**: Automatically resized to 32x32 pixels for CIFAR-10

## Development

### Adding New Models

To use a different model:

1. Replace `models/model_cifar10.h5` with your trained model
2. Update class names in `app/model.py`
3. Adjust image preprocessing if needed
4. Update the UI to reflect new classes

### Customizing the UI

The application uses Tailwind CSS for styling. Key files:

- `templates/index.html`: Upload interface
- `templates/result.html`: Results display
- Custom CSS in template `<style>` sections

## Notes

- Uploaded files are temporarily stored in `uploads/` directory
- Model files are stored in `models/` directory for better organization
- The application uses Indonesian language for user interface
- All error messages are localized to Indonesian
- Professional design without emojis for business use

## License

This project is for educational purposes. Please ensure you have proper licensing for any models or datasets used.
