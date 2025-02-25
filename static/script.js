document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const fileUpload = document.getElementById('file-upload');
    const textInput = document.getElementById('text-input');
    const summaryOutput = document.getElementById('summary-output');
    const summarizeBtn = document.getElementById('summarize-btn');
    const downloadSummaryBtn = document.getElementById('downloadSummaryBtn');
    const radioButtons = document.querySelectorAll('input[name="summary-type"]');

    // Ensure download button is disabled initially
    downloadSummaryBtn.disabled = true;

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

    document.getElementById('downloadCasefilesBtn').addEventListener('click', function() {
        // Replace with your file URL or logic to fetch casefiles
        const fileUrl = 'static/casefiles.zip';
        const link = document.createElement('a');
        link.href = fileUrl;
        link.download = 'casefiles.zip';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
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

        // Always disable download button when starting a new summary
        downloadSummaryBtn.disabled = true;

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
                // Keep download button disabled on error
                downloadSummaryBtn.disabled = true;
            } else {
                summaryOutput.value = data.summary;
                // Enable download button only if we have valid summary
                if (data.summary && data.summary.trim() !== '') {
                    downloadSummaryBtn.disabled = false;
                }
            }

        } catch (error) {
            console.error('Error:', error);
            summaryOutput.value = 'An error occurred while generating the summary. Please try again.';
            // Keep download button disabled on error
            downloadSummaryBtn.disabled = true;
        } finally {
            // Reset button state
            summarizeBtn.disabled = false;
            summarizeBtn.textContent = 'Summarize';
        }
    });

    // Add paste handler for convenience
    textInput.addEventListener('input', function() {
        // Clear any error messages that might be there
        if (textInput.value.startsWith('Error:') ||
            textInput.value.startsWith('Invalid file') ||
            textInput.value.startsWith('File too large')) {
            textInput.value = '';
        }

        // Reset download button state when input changes
        downloadSummaryBtn.disabled = true;
    });

    // Download summary button handler
    downloadSummaryBtn.addEventListener('click', function() {
        const summary = summaryOutput.value;
        // Extra check to make sure we have content to download
        if (summary && summary.trim() !== '' &&
            !summary.startsWith('Generating summary...') &&
            !summary.startsWith('Error:') &&
            !summary.startsWith('An error occurred')) {

            const blob = new Blob([summary], { type: 'text/plain' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'legal_summary.txt';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });

    // Prevent accidental navigation when there's unsaved text
    window.addEventListener('beforeunload', function(e) {
        if (textInput.value.trim() !== '') {
            e.preventDefault();
            e.returnValue = '';
        }
    });
});