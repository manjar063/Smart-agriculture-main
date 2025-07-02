import streamlit as st
from PIL import Image
import json
import tensorflow as tf
from utils.preprocessor import preprocess_image
from utils.disease_info import DISEASE_INFO
import os

def load_model_classes():
    # Load the trained model and class indices
    
    model=tf.keras.models.load_model('models/best_model.h5')
    with open('models/class_indices.json', 'r') as f:
        class_indices=json.load(f)
    class_names={ v: k for k , v in class_indices.items()}
    return model ,class_names
    
def predict_disease(image,model,class_names,confidence_threshold=0.7):
    # predict disease from image
    processed_image=preprocess_image(image)
    prediction=model.predict(processed_image)
    
    predicted_class_index = prediction.argmax()  # Get the index of the highest probability
    predicted_class = class_names[predicted_class_index]  # Get the class name
    confidence = float(prediction[0][predicted_class_index])  # Get the confidence score
    
    if confidence < confidence_threshold:
        return "Unknown", confidence
    
    predicted_class = class_names[predicted_class_index]
    return predicted_class, confidence

def main():
    st.set_page_config(page_title="Crop Disease Detection", layout="wide")
    
    st.title("🌿 Crop Disease Detection System")    
    st.write("Upload an image of a plant leaf to detect diseases and get treatment recommendations.")
    
    # load model and classes
    try:
        model, class_names = load_model_classes()  
    except Exception as e:
        st.error('Error loading model. Please make sure the model file exists.')
        return
        
    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
        # make prediction
        if st.button('Analyze Image'):
            with st.spinner('Analyzing...'):
                try:
                    disease, confidence = predict_disease(image, model ,class_names)
                    
                    with col2:
                        st.success(f"Disease Detected: {disease.replace('_', ' ')}")
                        st.progress(confidence) 
                        st.info(f"Confidence: {confidence:.2%}")
                        
                        # Display the disease name in the information section
                        st.subheader("Disease Information")
                        st.write(f"**Disease Name:** {disease.replace('_', ' ')}")  # Add this line
                
                    if disease in DISEASE_INFO:
                        info = DISEASE_INFO[disease]
                        
                        # Display disease information in expandable sections
                        with st.expander('Disease Information', expanded=True):
                            st.write(f"**Cause:**   {info['cause']}")
                            
                        with st.expander("Symptoms"):
                            for symptom in info['symptoms']:
                                st.write(f"• {symptom}")

                        with st.expander("Treatment Recommendations"):
                            for treatment in info['treatment']:
                                st.write(f"• {treatment}")

                        with st.expander("Prevention Measures"):
                            for prevention in info['prevention']:
                                st.write(f"• {prevention}")
                    else:
                        st.warning("Detailed information not available for this disease.")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    main()