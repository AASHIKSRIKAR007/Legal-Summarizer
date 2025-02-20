document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const fileUpload = document.getElementById('file-upload');
    const textInput = document.getElementById('text-input');
    const summaryOutput = document.getElementById('summary-output');
    const summarizeBtn = document.getElementById('summarize-btn');
    const radioButtons = document.querySelectorAll('input[name="summary-type"]');

    Swal.fire({
        title: "Welcome to Legal Document Summarization",
        html: `
            <p style="font-family:'Space Grotesque';text-align:center;font-size:16px;">
                This app  provides two types of summarization for legal documents using <strong>fine-tuned LegalBert and Bart models.</strong>
            </p>
            <p style="font-family:'Space Grotesque';text-align:center;font-size:16px;">
                <strong>Whole Summary:</strong> Generates a concise summary for the entire document.
            </p>
            <p style="font-family:'Space Grotesque';text-align:center;font-size:16px;">    
                <strong>Segmented Summary:</strong> Breaks the document into sections (Facts, Arguments, Analysis, Judgment, Statutes) and summarizes each separately.
            </p>
            <p style="font-family:'Space Grotesque';text-align:center;font-size:16px;">
                The models takes time to generate the summary.
            </p>
            <p style="font-family:'Space Grotesque';text-align:center;font-size:16px;">
                Upload txt casefile or paste your case file Click <strong>'Generate Summary'</strong> and wait for the results.
            </p>
            <p style="font-family:'Space Grotesque';text-align:center;font-size:16px;">
                Click <strong>'Generate Summary'</strong> and wait for the results.
            </p>
            <p style="font-family:'Space Grotesque';text-align:center;font-size:14px;">
                <strong>&copy; Thanuja Lksai Srikar</strong>
            </p>
        `,
        icon: "info",
        confirmButtonText: "OK",
        background: "#D3EAFF",
        width: "450px", // Adjust width
        padding: "1px",
        customClass: {
            title: "swal-title",
            content: "swal-content",
            confirmButton: "swal-confirm-button"
        }
    });

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