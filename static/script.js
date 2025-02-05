document.addEventListener("DOMContentLoaded", function() {
    const textInput = document.getElementById("text-input");
    const fileUpload = document.getElementById("file-upload");
    const generateBtn = document.getElementById("generate-btn");
    const summaryOutput = document.getElementById("summary-output");
    const radioButtons = document.querySelectorAll('input[name="summary-type"]');
  
    // When a .txt file is uploaded, read its contents and populate the text area.
    fileUpload.addEventListener("change", function(event) {
      const file = event.target.files[0];
      if (file && file.type === "text/plain") {
        const reader = new FileReader();
        reader.onload = function(e) {
          // Set the content of the text area to the file's content.
          textInput.value = e.target.result;
        };
        reader.readAsText(file);
      } else {
        alert("Please upload a valid .txt file");
      }
    });
  
    // When "Generate Summary" is clicked, use the text from the text area
    generateBtn.addEventListener("click", function() {
      // Get text from the text area (which could be directly entered text or file content).
      const textContent = textInput.value.trim();
  
      if (textContent === "") {
        alert("Please paste text or upload a .txt file before generating a summary.");
        return;
      }
  
      // Determine the selected summary type (whole or segmented).
      let summaryType = "whole-summary";
      radioButtons.forEach(rb => {
        if (rb.checked) {
          summaryType = rb.value;
        }
      });
  
      // Prepare the payload. Only the text content is sent.
      const payload = {
        text: textContent,
        summary_type: summaryType
      };
  
      // Send the payload as JSON via POST to the Flask API.
      fetch("/summarize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      })
        .then(response => response.json())
        .then(data => {
          if (data.error) {
            summaryOutput.value = "Error: " + data.error;
          } else if (data.summary) {
            summaryOutput.value = data.summary;
          }
        })
        .catch(error => {
          summaryOutput.value = "An error occurred: " + error;
        });
    });
  });
  