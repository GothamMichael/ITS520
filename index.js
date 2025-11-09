let modelInputCount = null;

async function updateExpectedInput() {
  const modelFile = document.getElementById("modelSelect").value;
  const infoDiv = document.getElementById("modelInfo");
  infoDiv.innerHTML = `<i>Loading model info for ${modelFile}...</i>`;

  try {
    const session = await ort.InferenceSession.create(modelFile + "?v=" + Date.now());
    const inputName = session.inputNames[0];
    const expectedShape = session.inputMetadata[inputName].dimensions;
    const expectedLen = expectedShape[1] || expectedShape[0] || 0;

    modelInputCount = expectedLen;
    infoDiv.innerHTML = `<b>Expected number of input values:</b> ${expectedLen}`;
  } catch (err) {
    modelInputCount = null;
    infoDiv.innerHTML = `<span style="color:red">⚠️ Couldn't load model info. ${err.message}</span>`;
  }
}

async function runModel() {
  const modelFile = document.getElementById("modelSelect").value;
  const rawInput = document.getElementById("inputValues").value;
  const outputDiv = document.getElementById("outputArea");

  outputDiv.innerHTML = `<i>Loading model: ${modelFile} ...</i>`;

  const inputs = rawInput
    .split(",")
    .map(v => parseFloat(v.trim()))
    .filter(v => !isNaN(v));

  if (inputs.length === 0) {
    outputDiv.innerHTML = `<span style="color:red">⚠️ Please enter valid numeric inputs.</span>`;
    return;
  }

  try {
    const session = await ort.InferenceSession.create(modelFile + "?v=" + Date.now());
    const inputName = session.inputNames[0];
    const expectedShape = session.inputMetadata[inputName].dimensions;
    const expectedLen = expectedShape[1] || inputs.length;

    let adjustedInputs = [...inputs];

    if (inputs.length < expectedLen) {
      while (adjustedInputs.length < expectedLen) adjustedInputs.push(0);
    } else if (inputs.length > expectedLen) {
      adjustedInputs = adjustedInputs.slice(0, expectedLen);
    }

    const tensor = new ort.Tensor("float32", new Float32Array(adjustedInputs), [1, expectedLen]);
    const results = await session.run({ [inputName]: tensor });
    const outputName = session.outputNames[0];
    const outputData = results[outputName].data;

    let html = `<b>✅ Model ran successfully!</b><br>`;
    html += `<b>Expected input length:</b> ${expectedLen}<br>`;
    html += `<b>Provided:</b> ${inputs.length} → Adjusted to match<br><br>`;
    html += `<b>Output tensor (${outputData.length} values):</b><br><br><table>`;
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

// Load expected input count when model is changed
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("modelSelect").addEventListener("change", updateExpectedInput);
  updateExpectedInput(); // run once on page load
});
