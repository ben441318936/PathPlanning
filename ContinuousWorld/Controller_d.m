%%
clear;

%%
r = 1;
L = 1;
I = 1;
a = 0.1;

Ts = 0.01;
A = [-a,0; 0,-a];
B = [r/2/I, r/2/I; r/I/L, -r/I/L];

sys = ss(Ts*A,Ts*B,eye(2),zeros(2),1);

%%
Q = eye(2);
R = eye(2);

[K,S,e] = lqr(sys,Q,R);

%%
eig(sys.A - sys.B*K)

%%
sys_control = ss(sys.A-sys.B*K,zeros(2),eye(2),zeros(2),1);
t = [0:1:10];
y = lsim(sys_control,zeros(length(t),2),t,[1,1]);

%%
figure()
subplot(2,1,1)
plot(t*Ts,y(:,1))
title("$$\omega$$","interpreter","latex");
subplot(2,1,2)
plot(t*Ts,y(:,2))
title("$$v$$","interpreter","latex");




