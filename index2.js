async function runModel() {
    const outputArea = document.getElementById('outputArea');
    outputArea.innerHTML = "Processing...";

    try {
        // 1. Load the model
        const session = await ort.InferenceSession.create('./spam_dl_model_full.onnx');

        // 2. Gather inputs from the labeled boxes
        const emailText = document.getElementById('email_text').value;
        const numWords = parseFloat(document.getElementById('num_words').value) || 0;
        const numLinks = parseFloat(document.getElementById('num_links').value) || 0;
        const exclamation = parseFloat(document.getElementById('exclamation_marks').value) || 0;
        const specialChars = parseFloat(document.getElementById('special_chars').value) || 0;

        // 3. Pre-processing (Tokenization)
        // Note: For a true Deep Learning model, 'emailText' needs to be converted 
        // to a sequence of numbers (tokens). Here is a placeholder for that tensor:
        const textSequence = tokenize(emailText); // You must define this based on your training
        const textTensor = new ort.Tensor('int32', Int32Array.from(textSequence), [1, textSequence.length]);

        // 4. Create Tensor for the numerical features
        const metaData = new Float32Array([numWords, numLinks, exclamation, specialChars]);
        const metaTensor = new ort.Tensor('float32', metaData, [1, 4]);

        // 5. Run inference
        const feeds = { 
            "text_input": textTensor, 
            "meta_input": metaTensor 
        };
        const results = await session.run(feeds);

        // 6. Display Result
        const prediction = results.output.data[0];
        const isSpam = prediction > 0.5;
        
        outputArea.innerHTML = isSpam 
            ? `<span style="color: red;">🚨 SPAM DETECTED (${(prediction * 100).toFixed(2)}%)</span>`
            : `<span style="color: green;">✅ HAM / SAFE (${((1 - prediction) * 100).toFixed(2)}%)</span>`;

    } catch (e) {
        outputArea.innerHTML = `<span style="color: orange;">Error: ${e.message}</span>`;
        console.error(e);
    }
}

// Simple placeholder for tokenization - this must match your model's vocabulary!
function tokenize(text) {
    // This is a dummy example. In a real scenario, you would map words to 
    // the specific integers used during your Python training.
    return text.toLowerCase().split(' ').map(word => word.length).slice(0, 50); 
}
