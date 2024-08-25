/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./flaskr/templates/**/*.html",
        "./flaskr/static/src/**/*.js",
        "node_modules/preline/dist/*.js",
        "node_modules/apexcharts/dist/*.js",
        "node_modules/apexcharts/dist/*.css",
        "node_modules/lodash/*.js",
    ],
    theme: {
        extend: {},
    },
    plugins: [
        require('@tailwindcss/forms'),
        require('preline/plugin'),
    ],
}

