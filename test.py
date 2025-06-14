#!/usr/bin/env python3
"""
Simple script to test your Gemini API key
Run this before running your main application
"""

import os
import sys

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed. Run: pip install google-generativeai")
    sys.exit(1)

def test_api_key(api_key):
    """Test if the API key works"""
    print(f"Testing API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else 'SHORT_KEY'}")
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Create a model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test with a simple prompt
        print("Sending test request...")
        response = model.generate_content("Say 'Hello, API is working!' in exactly those words.")
        
        print("✅ SUCCESS: API key is working!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: API key test failed")
        print(f"Error details: {str(e)}")
        
        if "API_KEY_INVALID" in str(e):
            print("\n🔧 SOLUTION: Your API key is invalid")
            print("1. Go to https://makersuite.google.com/app/apikey")
            print("2. Create a new API key")
            print("3. Replace your current key")
            
        elif "quota" in str(e).lower():
            print("\n🔧 SOLUTION: API quota exceeded")
            print("1. Check your usage at https://console.cloud.google.com/")
            print("2. Wait for quota reset or upgrade your plan")
            
        elif "permission" in str(e).lower():
            print("\n🔧 SOLUTION: Permission denied")
            print("1. Ensure Gemini API is enabled in your Google Cloud project")
            print("2. Check API key permissions")
            
        return False

def main():
    print("=== Gemini API Key Tester ===\n")
    
    # Try different ways to get the API key
    api_key = None
    
    # Method 1: Environment variable
    if os.getenv("GEMINI_API_KEY"):
        api_key = os.getenv("GEMINI_API_KEY")
        print("📝 Using API key from GEMINI_API_KEY environment variable")
    
    # Method 2: Direct input
    elif len(sys.argv) > 1:
        api_key = sys.argv[1]
        print("📝 Using API key from command line argument")
    
    # Method 3: Hardcoded (your current key)
    else:
        api_key = "AIzaSyAvzloY_NyX-yjtZb8EE_RdXPs3rPmMEso"
        print("📝 Using hardcoded API key")
    
    if not api_key:
        print("❌ No API key found!")
        print("Usage:")
        print("  python api_tester.py YOUR_API_KEY")
        print("  or set GEMINI_API_KEY environment variable")
        sys.exit(1)
    
    # Test the API key
    success = test_api_key(api_key)
    
    if success:
        print("\n🎉 Your API key is working! You can now run your Streamlit app.")
    else:
        print("\n💡 NEXT STEPS:")
        print("1. Get a new API key from: https://makersuite.google.com/app/apikey")
        print("2. Set it as environment variable: export GEMINI_API_KEY='your_new_key'")
        print("3. Or update your code with the new key")
        print("4. Run this tester again to verify")

if __name__ == "__main__":
    main()