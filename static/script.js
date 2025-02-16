document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const fileUpload = document.getElementById('file-upload');
    const textInput = document.getElementById('text-input');
    const summaryOutput = document.getElementById('summary-output');
    const summarizeBtn = document.getElementById('summarize-btn');
    const radioButtons = document.querySelectorAll('input[name="summary-type"]');

    // File upload handling
    fileUpload.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Create loading state
        textInput.value = 'Loading file...';

        const reader = new FileReader();

        reader.onload = function(e) {
            textInput.value = e.target.result;
        };

        reader.onerror = function() {
            textInput.value = 'Error reading file';
            console.error('File reading error');
        };

        // Check file type and size
        if (file.type === 'text/plain' ||
            file.type === 'application/pdf' ||
            file.type === 'application/msword' ||
            file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {

            if (file.size <= 5242880) { // 5MB limit
                reader.readAsText(file);
            } else {
                textInput.value = 'File too large. Please upload a file smaller than 5MB.';
            }
        } else {
            textInput.value = 'Invalid file type. Please upload a .txt, .doc, .docx, or .pdf file.';
        }
    });

    // Summarize button click handler
    summarizeBtn.addEventListener('click', async function() {
        const text = textInput.value.trim();

        if (!text) {
            alert('Please enter or upload some text to summarize.');
            return;
        }

        // Get selected summary type
        let summaryType = 'whole-summary'; // default
        radioButtons.forEach(radio => {
            if (radio.checked) {
                summaryType = radio.value;
            }
        });

        // Show loading state
        summarizeBtn.disabled = true;
        summarizeBtn.textContent = 'Summarizing...';
        summaryOutput.value = 'Generating summary...';

        try {
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    summary_type: summaryType
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                summaryOutput.value = `Error: ${data.error}`;
            } else {
                summaryOutput.value = data.summary;
            }

        } catch (error) {
            console.error('Error:', error);
            summaryOutput.value = 'An error occurred while generating the summary. Please try again.';
        } finally {
            // Reset button state
            summarizeBtn.disabled = false;
            summarizeBtn.textContent = 'Summarize';
        }
    });

    // Add paste handler for convenience
    textInput.addEventListener('paste', function() {
        // Clear any error messages that might be there
        if (textInput.value.startsWith('Error:') ||
            textInput.value.startsWith('Invalid file') ||
            textInput.value.startsWith('File too large')) {
            textInput.value = '';
        }
    });

    // Character count display (optional feature)
    textInput.addEventListener('input', function() {
        const wordCount = textInput.value.trim().split(/\s+/).filter(Boolean).length;
        const charCount = textInput.value.length;

        // You can add this to your HTML if you want to display counts
        // document.getElementById('word-count').textContent = `Words: ${wordCount}`;
        // document.getElementById('char-count').textContent = `Characters: ${charCount}`;
    });

    // Prevent accidental navigation when there's unsaved text
    window.addEventListener('beforeunload', function(e) {
        if (textInput.value.trim() !== '') {
            e.preventDefault();
            e.returnValue = '';
        }
    });
});