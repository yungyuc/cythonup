# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "debian/jessie64"

  # masterless, mount salt file root
  config.vm.synced_folder "salt/roots/", "/srv/"
  # config.vm.synced_folder "vm", "/home/vagrant"

  config.vm.provision :salt do |salt|
    salt.minion_config = "salt/minion"
    salt.run_highstate = true
  end

end

