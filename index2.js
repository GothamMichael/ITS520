async function runModel() {
    const outputArea = document.getElementById('outputArea');
    outputArea.innerHTML = "Fetching model files (this may take a moment)...";

    try {
        // 1. Fetch the .onnx structure
        const onnxResponse = await fetch('./spam_dl_model.onnx');
        const onnxBuffer = await onnxResponse.arrayBuffer();

        // 2. Fetch the .data weights
        const dataResponse = await fetch('./spam_dl_model.onnx.data');
        const dataBuffer = await dataResponse.arrayBuffer();

        // 3. Create the session with external data linked
        // We provide the weights as a Uint8Array in the 'externalData' object
        const session = await ort.InferenceSession.create(onnxBuffer, {
            externalData: [
                {
                    data: new Uint8Array(dataBuffer),
                    path: 'spam_dl_model.onnx.data' // This must match the name in the error
                }
            ]
        });

        outputArea.innerHTML = "Processing inference...";

        // 4. Gather Inputs (Matching your 4-feature vector)
        const numWords = parseFloat(document.getElementById('num_words').value) || 0;
        const numLinks = parseFloat(document.getElementById('num_links').value) || 0;
        const exclamation = parseFloat(document.getElementById('exclamation_marks').value) || 0;
        const specialChars = parseFloat(document.getElementById('special_chars').value) || 0;

        const inputData = new Float32Array([numWords, numLinks, exclamation, specialChars]);
        const inputTensor = new ort.Tensor('float32', inputData, [1, 4]);

        // 5. Run
        const results = await session.run({ "input": inputTensor });
        const prediction = results.output.data[0];

        outputArea.innerHTML = prediction > 0.5 
            ? `<b style="color:red">🚨 SPAM (${(prediction * 100).toFixed(1)}%)</b>`
            : `<b style="color:green">✅ HAM (${(prediction * 100).toFixed(1)}%)</b>`;

    } catch (e) {
        outputArea.innerHTML = `<span style="color: orange;">Error: ${e.message}</span>`;
        console.error("Manual Load Error:", e);
    }
}
