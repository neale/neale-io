# Neale.io

Simple website built with python, flask, html, css, and minimal JS. 

This repository doesn't contain all the images, but does contain some interesting components. 

* CPPN is a standalone webapp version of the popular model used to generate nice looking random generative art. Its pretty cheap to run on a small droplet and could be ripped out of the main code if anyone wanted to use it. By default this cppn uses random activations, but the pro version allows use of the fixed graph CPPN, the random activation version, and the WS random graph version as `Generate`, `Random`, and `Very Random` respectively.

* The gallery also looks fairly nice and could be ripped out to stand on its own. 

* The profile page is an adaptation from an old jekyll theme, itself adapted from [Shangtong Zhang's](https://shangtongzhang.github.io/) old github-pages site.

* `trials/` has some example outputs of the CPPN webapp

So I remember, I make changes in production, and then --

1. Check website status with `sudo systemctl status app`
2. Refresh website with `sudo systemctl restart app`

## Disclaimer

I'm no web developer, but if someone reads this and has suggestions particularly about the CPPN app and *particularly* the profile page, please open an issue or email me

