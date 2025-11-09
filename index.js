async function runModel() {
  const modelFile = document.getElementById("modelSelect").value;
  const rawInput = document.getElementById("inputValues").value;
  const outputDiv = document.getElementById("outputArea");

  // Parse input values
  let inputs = rawInput.split(",").map(v => parseFloat(v.trim())).filter(v => !isNaN(v));

  if (inputs.length === 0) {
    alert("Please enter at least one numeric input.");
    return;
  }

  try {
    outputDiv.innerHTML = `<i>Loading model ${modelFile} ...</i>`;

    // Load model
    const session = await ort.InferenceSession.create(modelFile + "?v=" + Date.now());

    // Create input tensor
    const tensor = new ort.Tensor("float32", new Float32Array(inputs), [1, inputs.length]);

    // Guess input name if not known
    const inputName = session.inputNames[0];

    // Run inference
    const results = await session.run({ [inputName]: tensor });
    const outputName = session.outputNames[0];
    const outputData = results[outputName].data;

    // Render results
    let html = `<b>Output tensor (${outputData.length} values):</b><br><table>`;
    for (let i = 0; i < outputData.length; i++) {
      html += `<tr><td>y[${i}]</td><td>${outputData[i].toFixed(4)}</td></tr>`;
    }
    html += "</table>";
    outputDiv.innerHTML = html;
  } catch (err) {
    console.error(err);
    outputDiv.innerHTML = `<span style="color:red">Error: ${err.message}</span>`;
  }
}
