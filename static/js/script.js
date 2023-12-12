const messagesContainer = document.getElementById('messages');
const uploadForm = document.getElementById('upload-form'); 
function uploadImage() {
    const formData = new FormData(uploadForm);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const { status, message } = data;

        const messageElement = document.createElement('div');
        messageElement.className = status === 'success' ? 'success' : 'error';
        messageElement.textContent = message;

        messagesContainer.innerHTML = '';
        messagesContainer.appendChild(messageElement);
    })
    .catch(error => {
        console.error('Error during upload:', error);
    });
} 

function previewImage(event) {
    const input = event.target;
    const previewImage = document.getElementById('image-preview');

    const file = input.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            previewImage.src = e.target.result;
        };

        reader.readAsDataURL(file);
    } else { 
        previewImage.src = defaultImage;
    }
}