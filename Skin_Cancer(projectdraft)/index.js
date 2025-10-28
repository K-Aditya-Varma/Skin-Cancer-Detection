document.getElementById('fileInput').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        // Display file name
        document.getElementById('fileInfo').innerHTML = `<p>Chosen File: ${file.name}</p>`;

        // Show image preview
        const reader = new FileReader();
        reader.onload = function (e) {
            const image = new Image();
            image.src = e.target.result;
            image.alt = "Uploaded Image";
            image.style.maxWidth = '100%';
            image.style.maxHeight = '300px';
            document.getElementById('imagePreview').innerHTML = '';
            document.getElementById('imagePreview').appendChild(image);
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    if (response.ok) {
        document.getElementById('result').innerHTML = `
            <p>Predicted Category: ${result.category}</p>
            <p>Predicted Stage: ${result.stage}</p>
        `;
    } else {
        document.getElementById('result').innerHTML = `
            <p>Error: ${result.error}</p>
        `;
    }
});
