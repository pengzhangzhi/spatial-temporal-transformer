# spatial-temporal-transformer

## Vscode connect to remote EC2 Instance with SSH

1. download SSH extensions in VScode

2. append local ssh public key `C:\Users\彭张智computational AI\.ssh\id_rsa.pub` to `.ssh/authorized_keys`
local id_rsa.pub:

ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDgrjjmmFh5AFzWKqw89cwYGgmCgiJGLcLKrYJ7qUPc6cfK/JEgBbHHftMvaIsVJp/JV/6uAuj/7MzPLaaHu46jAg1eFk1I8n7F8NnJwm/t5lrxu8tglcK6gnldMG4RRXLX1SfftzPeVjFPIRH6RzhTW3Z9G5WA+WQa9+m2e7+OqPYH5RUE35Y/rhvEk/Gk4yORgobdd8gaaMn3ODRjFyKspBCZIsvheJPtm2CMMvH1+PilTxTBIzrWzFNB7Ka6riBh9Ry3xz86Vnjp445IKSzlBYIAFOWhF1DoPxkffddAZJjRYAC5UuGp++PjK0Bp+TgM+RiYxH4cG9P9hQoVYRHh8t+X65+X/mkTQoJLDntLsSfYwCOpIhoceJ8vQA7hyNbEgytQePi0MHwb1reTduLggN4EVDwFezRrCdnEkDH/bb+g7HTCOfnhthL0twnwKRBbVBUukFuErp1NRQuDzGwsRxqgVEOWunQ2RSEy0Dg/AsuW3CegvjsXN3QQ8tIM9m8= 彭张智computational AI@peng-zhang-zhi-laptop

3. use vscode ssh directly connect to EC2 Instance.

format: 
ssh username@ip_address
e.g.:
ssh ubuntu@18.162.48.208 