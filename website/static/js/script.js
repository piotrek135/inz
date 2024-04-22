document.addEventListener('DOMContentLoaded', function() {
    var checkboxes = document.querySelectorAll('input[type="checkbox"]');

    checkboxes.forEach(function(checkbox) {
        checkbox.addEventListener('change', function() {
            updateStatus(checkbox);
        });

        var bird = checkbox.id;
        checkbox.checked = (birdsChoices[bird] === 1);
        console.log(birdsChoices[bird]);
    });

    function updateStatus(checkbox) {
        var bird = checkbox.id;
        console.log(bird)
        fetch(`/update_status/${bird}`, {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log(`Zaktualizowano status dla ${data.bird}: ${data.value}`);
            } else {
                console.error(`Błąd podczas aktualizacji statusu: ${data.message}`);
            }
        })
        .catch(error => console.error('Błąd fetch:', error));
    }

});