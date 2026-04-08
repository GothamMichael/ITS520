async function runModel() {
    const outputArea = document.getElementById('outputArea');
    outputArea.innerHTML = "Initializing model...";

    try {
        const ts = new Date().getTime();
        const modelUrl = `./spam_dl_model.onnx?v=${ts}`; // Verified filename from your export
        
        const session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm']
        });

        outputArea.innerHTML = "Analyzing inputs...";

        const featureOrder = [
            'num_words', 'num_characters', 'num_exclamation_marks', 'num_links',
            'has_suspicious_link', 'num_attachments', 'has_attachment', 
            'sender_reputation_score', 'email_hour', 'email_day_of_week', 
            'is_weekend', 'num_recipients', 'contains_money_terms', 'contains_urgency_terms'
        ];

        const featureValues = featureOrder.map(id => {
            let val = parseFloat(document.getElementById(id).value) || 0;
            
            if (id === 'sender_reputation_score') {
                val = val / 100;
            }
            
            return val;
        });

        const inputTensor = new ort.Tensor('float32', new Float32Array(featureValues), [1, 14]);

        // 6. Run Inference
        const results = await session.run({ "input": inputTensor });
        const outputData = results.output.data; // Expected format: [Ham_Logit, Spam_Logit]

        // --- STABILITY FIX: SOFTMAX CALCULATION ---
        // We subtract the max value to prevent Infinity errors in Math.exp
        const maxLogit = Math.max(outputData[0], outputData[1]);
        const expHam = Math.exp(outputData[0] - maxLogit);
        const expSpam = Math.exp(outputData[1] - maxLogit);
        const probSpam = expSpam / (expHam + expSpam);
        // ------------------------------------------

        const isSpam = probSpam > 0.5;
        const confidence = (isSpam ? probSpam : 1 - probSpam) * 100;

        // Realistic capping: don't show 100% unless it's truly absolute
        const displayConf = confidence > 99.99 ? 99.99 : confidence;

        outputArea.innerHTML = isSpam 
            ? `<div style="color: #d93025;"><strong>SPAM DETECTED</strong><br>Confidence: ${displayConf.toFixed(2)}%</div>`
            : `<div style="color: #188038;"><strong>REAL EMAIL DETECTED</strong><br>Confidence: ${displayConf.toFixed(2)}%</div>`;

    } catch (e) {
        outputArea.innerHTML = `<span style="color: orange;">Error: ${e.message}</span>`;
        console.error("Inference Error:", e);
    }
}
