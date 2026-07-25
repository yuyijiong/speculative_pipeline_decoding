

q=[B,8,H,d]
k=[B,8,H,d]
v=[B,8,H,d]

k,v=kv_cache.update(k,v)

k,v=[B,S,H,d]

q,k=pe(q,k)


attn_weights=q@k.transpose(-2,-1)/sqrt(d)

attn_weights=[B,H,8,S]

computation=B*H*8*S*d*2

attn_weights=softmax(attn_weights,dim=-1)

out=attn_weights@v





