eval "$(starship init bash)"
export PATH=$HOME/.local/bin:$PATH
export CPM_SOURCE_CACHE=$HOME/.cache/CPM
export PATH=$HOME/.cargo/bin:$PATH
eval "$(fzf --bash)"
source ~/fzf-git.sh/fzf-git.sh
export FZF_CTRL_T_OPTS="--preview 'bat -n --color=always --line-range :500 {}'"
export FZF_ALT_C_OPTS="--preview 'lsd --tree --color=always {} | head -200'"
