+images = {};   % create an empty cell array
k = 1;         % index into cell array for next read image

startfile=1; 
endfile=3;
startlevel=0;
endlevel=2;
%load animate images
for i=startfile:endfile;
    for s=startlevel:endlevel;

          % read the image into the kth position of the cell array
          images{k}=imread(['ani_',num2str(i),'_level_',num2str(s),'.png']);
          % increment k for the next iteration
          k = k + 1;

      end
  end