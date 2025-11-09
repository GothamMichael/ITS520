async function runModel() {
  const modelFile = document.getElementById("modelSelect").value;
  const rawInput = document.getElementById("inputValues").value;
  const outputDiv = document.getElementById("outputArea");

  // Show loading message
  outputDiv.innerHTML = `<i>Loading model: ${modelFile} ...</i>`;

  // Parse inputs
  const inputs = rawInput
    .split(",")
    .map(v => parseFloat(v.trim()))
    .filter(v => !isNaN(v));

  if (inputs.length === 0) {
    outputDiv.innerHTML = `<span style="color:red">⚠️ Please enter valid numeric inputs.</span>`;
    return;
  }

  try {
    // Load ONNX model
    const session = await ort.InferenceSession.create(modelFile + "?v=" + Date.now());

    // Determine model input name dynamically
    const inputName = session.inputNames[0];
    const tensor = new ort.Tensor("float32", new Float32Array(inputs), [1, inputs.length]);

    // Run inference
    const results = await session.run({ [inputName]: tensor });
    const outputName = session.outputNames[0];
    const outputData = results[outputName].data;

    // Render nicely
    let html = `<b>✅ Model ran successfully!</b><br><br>`;
    html += `<b>Output tensor (${outputData.length} values):</b><br><br>`;
    html += `<table>`;
    for (let i = 0; i < outputData.length; i++) {
      html += `<tr><td>Output[${i}]</td><td>${outputData[i].toFixed(4)}</td></tr>`;
    }
    html += `</table>`;
    outputDiv.innerHTML = html;
  } catch (err) {
    console.error(err);
    outputDiv.innerHTML = `<span style="color:red">❌ Error running model:<br>${err.message}</span>`;
  }
}
