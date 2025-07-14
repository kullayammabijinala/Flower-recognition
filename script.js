// script.js (Only the relevant identifyButton.addEventListener part is shown for brevity)

document.addEventListener('DOMContentLoaded', () => {
    // ... (Your existing element selections and image upload event listener) ...

    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const imagePlaceholder = document.getElementById('imagePlaceholder');
    const identifyButton = document.getElementById('identifyButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorMessage = document.getElementById('errorMessage');
    const resultBox = document.getElementById('resultBox');
    const predictionResult = document.getElementById('predictionResult');
    const confidenceResult = document.getElementById('confidenceResult');
    const allPredictionsList = document.getElementById('allPredictionsList');

    // Function to handle image preview and reset UI on new image selection
    imageUpload.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                imagePlaceholder.style.display = 'none';

                errorMessage.textContent = '';
                resultBox.style.display = 'none';
                predictionResult.textContent = '';
                confidenceResult.textContent = '';
                allPredictionsList.innerHTML = '';
            };
            reader.readAsDataURL(file);
        } else {
            imagePreview.src = '#';
            imagePreview.style.display = 'none';
            imagePlaceholder.style.display = 'block';

            errorMessage.textContent = '';
            resultBox.style.display = 'none';
            predictionResult.textContent = '';
            confidenceResult.textContent = '';
            allPredictionsList.innerHTML = '';
        }
    });


    // Function to handle the "Identify Flower" button click
    identifyButton.addEventListener('click', async () => {
        const file = imageUpload.files[0];
        if (!file) {
            errorMessage.textContent = 'Please select an image first.';
            errorMessage.style.color = 'red';
            return;
        }

        loadingIndicator.style.display = 'flex';
        identifyButton.disabled = true;
        errorMessage.textContent = '';
        resultBox.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);
// ... (existing script.js code above the fetch call) ...

        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                body: formData,
            });

            let data; // Declare 'data' variable here

            // Determine if the response is JSON or plain text based on Content-Type header
            const contentType = response.headers.get('content-type');

            if (contentType && contentType.includes('application/json')) {
                // If it's JSON, try to parse it
                try {
                    data = await response.json();
                } catch (e) {
                    // If JSON parsing itself fails, it's a corrupted/unexpected JSON
                    console.error("JSON parsing error:", e);
                    // Provide a fallback error message
                    data = { error: "Server response was corrupted JSON. Please check backend logs." };
                }
            } else {
                // If it's not JSON (e.g., HTML for 404/500), read as text
                const textResponse = await response.text();
                console.warn("Server responded with non-JSON content:", textResponse);
                // Create a generic error object from the text response
                data = { error: `Server responded with unexpected content (Status: ${response.status}): ${textResponse.substring(0, 150)}...` };
            }

            // Now, handle the response based on the HTTP status (response.ok)
            // and the 'data' object we just parsed
            if (!response.ok) {
                // If HTTP status is an error (e.g., 400, 500)
                const detailedError = data.error || `Unknown server error (Status: ${response.status})`;
                throw new Error(`Prediction failed: ${detailedError}`);
            }

            // If we made it here, response.ok is true, and data should contain prediction results
            console.log('Prediction data from backend:', data);

            if (data.prediction) {
                predictionResult.textContent = `Species: ${data.prediction}`;
                confidenceResult.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
                resultBox.style.display = 'block';

                allPredictionsList.innerHTML = '';
                if (data.all_predictions && typeof data.all_predictions === 'object') {
                    for (const className in data.all_predictions) {
                        if (data.all_predictions.hasOwnProperty(className)) {
                            const probability = data.all_predictions[className];
                            const listItem = document.createElement('li');
                            listItem.textContent = `${className}: ${probability.toFixed(2)}%`;
                            allPredictionsList.appendChild(listItem);
                        }
                    }
                }
            } else if (data.error) {
                // If backend successfully returned JSON, but it contains an 'error' key
                errorMessage.textContent = `Error: ${data.error}`;
                errorMessage.style.color = 'red';
            } else {
                errorMessage.textContent = 'Could not get a prediction. Unexpected response format.';
                errorMessage.style.color = 'red';
            }

        } catch (error) {
            // Catch any network errors or errors thrown by our logic above
            console.error('Error identifying flower:', error);
            errorMessage.textContent = `Error: ${error.message}. Please try again.`;
            errorMessage.style.color = 'red';
        } finally {
            loadingIndicator.style.display = 'none';
            identifyButton.disabled = false;
        }
    });
});