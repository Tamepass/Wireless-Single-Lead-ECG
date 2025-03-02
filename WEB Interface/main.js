const menuIcon = document.querySelector('#menu-icon');
const navbar = document.querySelector('.navbar');

// menuIcon.onclick = () => {
//     menuIcon.classList.toggle('bx-x');
//     navbar.classList.toggle('active');
// }

function toggleDropdown(id) {
    let dropdown = document.getElementById(id);
    let allDropdowns = document.querySelectorAll(".dropdown-content");
    
    allDropdowns.forEach((drop) => {
        if (drop.id !== id) {
            drop.classList.remove("show");
        }
    });

    dropdown.classList.toggle("show");
}

// Close dropdown when clicking outside
window.onclick = function(event) {
    if (!event.target.matches('.dropdown a')) {
        document.querySelectorAll('.dropdown-content').forEach(drop => {
            drop.classList.remove('show');
        });
    }
};
