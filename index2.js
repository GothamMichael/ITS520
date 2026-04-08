async function runModel() {
    const outputArea = document.getElementById('outputArea');
    outputArea.innerHTML = "Initializing model...";

    try {
        // 1. Fetch only the .onnx file with cache-busting
        const ts = new Date().getTime();
        const modelUrl = `./spam_dl_model.onnx?v=${ts}`;
        
        // 2. Create session directly from the URL
        const session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm']
        });

        outputArea.innerHTML = "Model Loaded. Analyzing inputs...";

        // 3. Exact order of the 14 features
        const featureOrder = [
            'num_words', 'num_characters', 'num_exclamation_marks', 'num_links',
            'has_suspicious_link', 'num_attachments', 'has_attachment', 
            'sender_reputation_score', 'email_hour', 'email_day_of_week', 
            'is_weekend', 'num_recipients', 'contains_money_terms', 'contains_urgency_terms'
        ];

        // 4. Map values
        const featureValues = featureOrder.map(id => {
            return parseFloat(document.getElementById(id).value) || 0;
        });

        // 5. Create Tensor [1, 14]
        const inputTensor = new ort.Tensor('float32', new Float32Array(featureValues), [1, 14]);

        // 6. Run Inference
        const results = await session.run({ "input": inputTensor });
        const outputData = results.output.data; 

        // Calculation: If your model outputs 2 numbers (logits), we convert them to %
        // If it only outputs 1 number, use: const prob = outputData[0];
        const prob = Math.exp(outputData[1]) / (Math.exp(outputData[0]) + Math.exp(outputData[1]));
        
        // 7. Display Results
        const isSpam = prob > 0.5;
        const confidence = (isSpam ? prob : 1 - prob) * 100;

        outputArea.innerHTML = isSpam 
            ? `<div style="color: #d93025;"><strong>🚨 SPAM DETECTED</strong><br>Confidence: ${confidence.toFixed(2)}%</div>`
            : `<div style="color: #188038;"><strong>✅ HAM (SAFE)</strong><br>Confidence: ${confidence.toFixed(2)}%</div>`;

    } catch (e) {
        outputArea.innerHTML = `<span style="color: orange;">Error: ${e.message}</span>`;
        console.error("Inference Error:", e);
    }
}
