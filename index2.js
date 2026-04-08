async function runModel() {
    const outputArea = document.getElementById('outputArea');
    outputArea.innerHTML = "Fetching model files...";

    try {
        // 1. Fetch files with cache busting
        const timestamp = new Date().getTime();
        const [onnxRes, dataRes] = await Promise.all([
            fetch(`./spam_dl_model.onnx?v=${timestamp}`),
            fetch(`./spam_dl_model.onnx.data?v=${timestamp}`)
        ]);

        if (!onnxRes.ok || !dataRes.ok) {
            throw new Error("Could not find the model files on the server.");
        }

        const onnxBuffer = await onnxRes.arrayBuffer();
        const dataBuffer = await dataRes.arrayBuffer();

        // 2. Configure Session
        // Note: 'externalData' MUST match the 'path' the .onnx file is looking for.
        const sessionOptions = {
            executionProviders: ['wasm'], 
            externalData: [
                {
                    data: new Uint8Array(dataBuffer),
                    path: 'spam_dl_model.onnx.data' 
                }
            ]
        };

        // 3. Create Session
        // We pass the model buffer first, then the options object
        const session = await ort.InferenceSession.create(onnxBuffer, sessionOptions);

        outputArea.innerHTML = "Running inference...";

        // 4. Inputs
        const features = [
            parseFloat(document.getElementById('num_words').value) || 0,
            parseFloat(document.getElementById('num_links').value) || 0,
            parseFloat(document.getElementById('exclamation_marks').value) || 0,
            parseFloat(document.getElementById('special_chars').value) || 0
        ];

        const inputTensor = new ort.Tensor('float32', new Float32Array(features), [1, 4]);

        // 5. Execute
        // Ensure "input" matches your input_names=['input'] from the Python script
        const results = await session.run({ "input": inputTensor });
        
        // Ensure "output" matches your output_names=['output']
        const prediction = results.output.data[0];

        outputArea.innerHTML = prediction > 0.5 
            ? `<b style="color:red">🚨 SPAM (${(prediction * 100).toFixed(1)}%)</b>`
            : `<b style="color:green">✅ HAM (${(prediction * 100).toFixed(1)}%)</b>`;

    } catch (e) {
        outputArea.innerHTML = `<span style="color: orange;">Error: ${e.message}</span>`;
        console.error("Critical Failure:", e);
    }
}
