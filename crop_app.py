from crop_recommendation_model import CropRecommendationModel

def main():
    print("="*60)
    print("    AI-Based Crop Recommendation System")
    print("="*60)
    
    # Initialize the model
    model = CropRecommendationModel()
    
    # Load and train the model
    print("\n[INFO] Loading dataset and training model...")
    df = model.load_and_preprocess_data("D:\\Ankita Project\\Crop_recommendation.csv")
    model.train_models(df)
    print("[INFO] Model trained successfully!\n")
    
    # Get user inputs
    print("Enter the following soil and climate details:\n")
    
    try:
        n = float(input("Nitrogen (N) content: "))
        p = float(input("Phosphorus (P) content: "))
        k = float(input("Potassium (K) content: "))
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        ph = float(input("pH value: "))
        rainfall = float(input("Rainfall (mm): "))
        
        # Get prediction
        print("\n[INFO] Analyzing your inputs...\n")
        result = model.predict_crop_and_variety(n, p, k, temperature, humidity, ph, rainfall)
        
        # Display results
        print("="*60)
        print("         RECOMMENDATION RESULTS")
        print("="*60)
        print(f"\n🌾 Recommended Crop: {result['crop'].upper()}")
        print(f"🌱 Recommended Variety: {result['variety'].upper()}")
        print(f"✅ Crop Confidence: {result['crop_confidence']:.2f}%")
        print(f"✅ Variety Confidence: {result['variety_confidence']:.2f}%")
        
        print("\n" + "="*60)
        print("         EXPLANATION")
        print("="*60)
        print(result['explanation']['recommendation_summary'])
        
        print("\n📊 Why this crop?")
        print(result['explanation']['why_this_crop'])
        
        print("\n💡 Cultivation Tips:")
        for tip in result['explanation']['cultivation_tips']:
            print(f"  • {tip}")
        
        print("\n" + "="*60)
        
    except ValueError:
        print("\n❌ ERROR: Please enter valid numeric values!")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

if __name__ == "__main__":
    main()
