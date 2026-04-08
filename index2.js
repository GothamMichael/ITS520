async function runModel() {
    const outputArea = document.getElementById('outputArea');
    outputArea.innerHTML = "Initializing model and weights...";

    try {
        // 1. Fetch .onnx and .data files with cache-busting
        const ts = new Date().getTime();
        const [onnxRes, dataRes] = await Promise.all([
            fetch(`./spam_dl_model.onnx?v=${ts}`),
            fetch(`./spam_dl_model.onnx.data?v=${ts}`)
        ]);

        if (!onnxRes.ok || !dataRes.ok) throw new Error("Model files not found on server.");

        const onnxBuffer = await onnxRes.arrayBuffer();
        const dataBuffer = await dataRes.arrayBuffer();

        // 2. Build the session by manually mounting the data buffer
        const session = await ort.InferenceSession.create(onnxBuffer, {
            executionProviders: ['wasm'],
            externalData: [
                {
                    data: new Uint8Array(dataBuffer),
                    path: 'spam_dl_model.onnx.data' 
                }
            ]
        });

        outputArea.innerHTML = "Model Loaded. Analyzing inputs...";

        // 3. Define the exact order of the 14 features from your training list
        const featureOrder = [
            'num_words', 'num_characters', 'num_exclamation_marks', 'num_links',
            'has_suspicious_link', 'num_attachments', 'has_attachment', 
            'sender_reputation_score', 'email_hour', 'email_day_of_week', 
            'is_weekend', 'num_recipients', 'contains_money_terms', 'contains_urgency_terms'
        ];

        // 4. Map HTML values to a Float32Array
        const featureValues = featureOrder.map(id => {
            const val = document.getElementById(id).value;
            return parseFloat(val) || 0;
        });

        // 5. Create Tensor [1, 14]
        const inputTensor = new ort.Tensor('float32', new Float32Array(featureValues), [1, 14]);

        // 6. Run Inference
        const results = await session.run({ "input": inputTensor });
        const prediction = results.output.data[0];
        
        // 7. Display Results
        const isSpam = prediction > 0.5;
        const confidence = (isSpam ? prediction : 1 - prediction) * 100;

        outputArea.innerHTML = isSpam 
            ? `<div style="color: #d93025;"><strong>🚨 SPAM DETECTED</strong><br>Confidence: ${confidence.toFixed(2)}%</div>`
            : `<div style="color: #188038;"><strong>✅ HAM (SAFE)</strong><br>Confidence: ${confidence.toFixed(2)}%</div>`;

    } catch (e) {
        outputArea.innerHTML = `<span style="color: orange;">Error: ${e.message}</span>`;
        console.error("Inference Error:", e);
    }
}
