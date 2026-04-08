async function runModel() {
    const outputArea = document.getElementById('outputArea');
    outputArea.innerHTML = "Processing...";

    try {
        // 1. Load the model - Using a cache-busting suffix to force reload
        const modelPath = './spam_dl_model.onnx?v=' + new Date().getTime();
        const session = await ort.InferenceSession.create(modelPath);

        // 2. Gather inputs
        const numWords = parseFloat(document.getElementById('num_words').value) || 0;
        const numLinks = parseFloat(document.getElementById('num_links').value) || 0;
        const exclamation = parseFloat(document.getElementById('exclamation_marks').value) || 0;
        const specialChars = parseFloat(document.getElementById('special_chars').value) || 0;

        // 3. Create the input tensor 
        // Based on your Python export: input_names=['input']
        // We create a float32 array of the 4 features
        const inputData = new Float32Array([numWords, numLinks, exclamation, specialChars]);
        const inputTensor = new ort.Tensor('float32', inputData, [1, 4]);

        // 4. Run inference
        const feeds = { "input": inputTensor };
        const results = await session.run(feeds);

        // 5. Display Result
        // Use the output name from your Python script: output_names=['output']
        const prediction = results.output.data[0];
        const isSpam = prediction > 0.5;
        
        outputArea.innerHTML = isSpam 
            ? `<span style="color: #d93025; font-size: 24px;">🚨 SPAM DETECTED</span><br>Probability: ${(prediction * 100).toFixed(2)}%`
            : `<span style="color: #188038; font-size: 24px;">✅ HAM / SAFE</span><br>Probability: ${(prediction * 100).toFixed(2)}%`;

    } catch (e) {
        outputArea.innerHTML = `<span style="color: orange;">Error: ${e.message}</span>`;
        console.error("Full Error Object:", e);
    }
}
